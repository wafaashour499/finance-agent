import re
from datetime import datetime
import pandas as pd
from pathlib import Path

MAX_REJECTION_RATE = 40

UNIFIED_SCHEMA = {
    "date": "Date",
    "amount": "Amount",
    "category": "Category",
    "description": "Description",
    "type": "Type",
}

COLUMN_ALIASES = {
    "date": [
        "date",
        "transaction_date",
        "transaction date",
        "trans_date",
        "datetime",
    ],
    "amount": [
        "amount",
        "transaction_amount",
        "transaction amount",
        "value",
        "total",
    ],
    "category": [
        "category",
        "expense_category",
        "expense category",
        "transaction_category",
    ],
    "description": [
        "description",
        "details",
        "merchant",
        "transaction_description",
        "note",
        "notes",
    ],
    "type": [
        "type",
        "transaction_type",
        "transaction type",
        "income_expense",
        "direction",
    ],
}

def load_csv(file_path):
    """Load a CSV file without modifying the original."""
    for encoding in ["utf-8", "cp1256", "latin1"]:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Unable to read CSV encoding.")

def normalize_column_names(df):
    """Normalize column names for matching."""
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    return df

def map_columns(df):
    """Map different source column names to the unified schema."""
    df = normalize_column_names(df)
    column_mapping = {}
    used_columns = set()
    for standard_name, aliases in COLUMN_ALIASES.items():
        normalized_aliases = {
            alias.lower().replace(" ", "_") for alias in aliases
        }
        matches = [
            column for column in df.columns if column in normalized_aliases
        ]
        if len(matches) > 1:
            raise ValueError(
                f"Multiple columns match '{standard_name}': {matches}"
            )
        if matches:
            source_column = matches[0]
            column_mapping[source_column] = UNIFIED_SCHEMA[standard_name]
            used_columns.add(source_column)
        elif standard_name in ["date", "amount", "type"]:
            raise ValueError(
                f"Could not identify required column: {standard_name}. "
                f"Available columns: {list(df.columns)}"
            )
    df = df.rename(columns=column_mapping)
    # Optional columns receive default values.
    for column in ["Category", "Description"]:
        if column not in df.columns:
            df[column] = "Unknown"
    return df[list(UNIFIED_SCHEMA.values())].copy()

def parse_date_value(value, day_first=None):
    """Parse dates without guessing ambiguous day/month formats."""
    if pd.isna(value) or str(value).strip() == "":
        return pd.NaT

    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value)

    value = str(value).strip()

    formats = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
    ]

    for fmt in formats:
        try:
            return pd.Timestamp(datetime.strptime(value, fmt))
        except ValueError:
            continue

    # Slash or dash formats: DD/MM/YYYY or MM/DD/YYYY
    match = re.fullmatch(r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})", value)

    if match:
        first, second, year = map(int, match.groups())

        if first > 12:
            day, month = first, second
        elif second > 12:
            month, day = first, second
        elif day_first is None:
            raise ValueError(
                f"Ambiguous date: {value}. "
                "Please specify whether the format is DD/MM/YYYY "
                "or MM/DD/YYYY."
            )
        elif day_first:
            day, month = first, second
        else:
            month, day = first, second

        try:
            return pd.Timestamp(year=year, month=month, day=day)
        except ValueError:
            return pd.NaT

    return pd.to_datetime(value, errors="coerce")

def parse_amount_series(series):
    """Parse amounts while preserving numeric accuracy."""
    values = series.astype("string").str.strip()
    currency_patterns = {
        "USD": r"\$|\bUSD\b",
        "EUR": r"€|\bEUR\b",
        "GBP": r"£|\bGBP\b",
        "EGP": r"ج\.?م\.?|\bEGP\b",
    }
    detected_currencies = set()
    for currency, pattern in currency_patterns.items():
        if values.str.contains(
            pattern, case=False, regex=True, na=False
        ).any():
            detected_currencies.add(currency)
    if len(detected_currencies) > 1:
        raise ValueError(
            "Multiple currencies detected in the same file: "
            f"{sorted(detected_currencies)}. "
            "Convert all amounts to one currency before analysis."
        )

    def parse_one(value):
        if pd.isna(value) or value == "":
            return float("nan")
        value = str(value).strip()
        value = re.sub(
            r"[$€£]|ج\.?م\.?|\b(?:USD|EUR|GBP|EGP)\b", "", value, flags=re.IGNORECASE
        )
        value = value.replace(" ", "")
        if "," in value and "." in value:
            if not re.fullmatch(
                r"[+-]?\d{1,3}(,\d{3})+(\.\d+)?", value
            ):
                return float("nan")
            value = value.replace(",", "")
        elif "," in value:
            if not re.fullmatch(
                r"[+-]?\d{1,3}(,\d{3})+", value
            ):
                return float("nan")
            value = value.replace(",", "")
        try:
            return float(value)
        except ValueError:
            return float("nan")
    return values.apply(parse_one)

def prepare_csv(file_path):
    """Load CSV data and map it to the unified schema."""
    raw_df = load_csv(file_path)
    return map_columns(raw_df)

# الخطوة 1: تحديث تعريف الدالة لتقبل المعامل date_format
def validate_and_clean(df, date_format=None):
    """Validate transactions and generate a detailed quality report."""
    original_rows = len(df)
    # Possible duplicates are reported, not removed (repeated purchases are legitimate).
    duplicate_count = int(df.duplicated(keep="first").sum())
    raw_df = df.copy()
    # Keep original values for accurate validation and reporting.
    working_df = raw_df.copy()
    error_counts = {
        "missing_date": 0,
        "invalid_date": 0,
        "future_date": 0,
        "missing_amount": 0,
        "non_numeric_amount": 0,
        "negative_amount": 0,
        "zero_amount": 0,
        "missing_type": 0,
        "invalid_type": 0,
    }
    # Track genuinely missing categories and descriptions.
    missing_category_mask = (
        working_df["Category"].isna() | working_df["Category"].astype("string").str.strip().eq("").fillna(False)
    )
    missing_description_mask = (
        working_df["Description"].isna() | working_df["Description"].astype("string").str.strip().eq("").fillna(False)
    )
    missing_categories_filled = int(missing_category_mask.sum())
    missing_descriptions_filled = int(missing_description_mask.sum())
    # Normalize text fields.
    for column in ["Category", "Description", "Type"]:
        working_df[column] = (
            working_df[column].astype("string").str.strip()
        )
    working_df["Category"] = (
        working_df["Category"]
        .replace("", pd.NA)
        .fillna("Unknown")
        .str.strip()
        .str.casefold()
        .str.title()
    )
    working_df["Description"] = (
        working_df["Description"].replace("", pd.NA).fillna("Unknown")
    )
    working_df["Type"] = working_df["Type"].str.lower()

    # الخطوة 2: معالجة التواريخ بناءً على الشرط المطلوب في الصورة والملاحظة
    if date_format:
        parsed_dates = pd.to_datetime(working_df["Date"], format=date_format, errors="coerce")
    else:
        day_first = detect_day_first(working_df["Date"])
        parsed_dates = working_df["Date"].apply(
            lambda v: parse_date_value(v, day_first)
        )

    amount_values = working_df["Amount"].astype("string").str.strip()
    parsed_amounts = parse_amount_series(working_df["Amount"])
    # Build rejection reasons while preserving original row values.
    row_reasons = []
    today = pd.Timestamp.today().normalize()
    for position, index in enumerate(working_df.index):
        reasons = []
        # تم إصلاح الخطأ الإملائي هنا واستبدال المتغير غير المعرف بـ parsed_dates
        date_missing = (
            pd.isna(working_df.at[index, "Date"])
            or str(working_df.at[index, "Date"]).strip() == ""
        )
        if date_missing:
            reasons.append("Missing date")
            error_counts["missing_date"] += 1
        elif pd.isna(parsed_dates.loc[index]):
            reasons.append("Invalid date")
            error_counts["invalid_date"] += 1
        elif parsed_dates.loc[index] > today:
            reasons.append("Future date")
            error_counts["future_date"] += 1
        amount_missing = (
            pd.isna(working_df.at[index, "Amount"]) or pd.isna(amount_values.loc[index]) or amount_values.loc[index] == ""
        )
        if amount_missing:
            reasons.append("Missing amount")
            error_counts["missing_amount"] += 1
        elif pd.isna(parsed_amounts.loc[index]):
            reasons.append("Non-numeric amount")
            error_counts["non_numeric_amount"] += 1
        elif parsed_amounts.loc[index] < 0:
            reasons.append("Negative amount (possible refund)")
            error_counts["negative_amount"] += 1
        elif parsed_amounts.loc[index] == 0:
            reasons.append("Zero amount")
            error_counts["zero_amount"] += 1
        transaction_type = working_df.at[index, "Type"]
        if pd.isna(transaction_type) or transaction_type == "":
            reasons.append("Missing transaction type")
            error_counts["missing_type"] += 1
        elif transaction_type not in ["income", "expense"]:
            reasons.append("Invalid transaction type")
            error_counts["invalid_type"] += 1
        row_reasons.append(reasons)
        # Separate valid and rejected rows.
    rejected_mask = pd.Series(
        [bool(reasons) for reasons in row_reasons], index=working_df.index
    )
    rejected_df = raw_df.loc[rejected_mask].copy()
    rejected_df["Rejection_Reasons"] = [
        "; ".join(reasons) for reasons in row_reasons if reasons
    ]
    clean_df = working_df.loc[~rejected_mask].copy()
    
    # Apply validated data types.
    clean_df["Date"] = parsed_dates.loc[~rejected_mask]
    clean_df["Amount"] = parsed_amounts.loc[~rejected_mask]
    
    rejected_count = int(rejected_mask.sum())
    total_rows = original_rows
    rejection_rate = (
        rejected_count / total_rows * 100 if total_rows > 0 else 0
    )
    processing_status = (
        "STOPPED" if rejection_rate > MAX_REJECTION_RATE else "SUCCESS"
    )
    valid_rows = len(clean_df)
    
    quality_report = {
        "original_rows": original_rows,
        "possible_duplicate_rows": duplicate_count,
        "rejected_rows": rejected_count,
        "accepted_rows": valid_rows,
        "rejection_rate": round(rejection_rate, 2),
        "missing_categories_filled": missing_categories_filled,
        "missing_descriptions_filled": missing_descriptions_filled,
        "error_counts": error_counts,
        "processing_status": processing_status,
    }
    
    # Block all downstream analysis if rejection exceeds 40%.
    if processing_status == "STOPPED":
        quality_report["valid_rows_blocked"] = valid_rows
        quality_report["accepted_rows"] = 0
        clean_df = clean_df.iloc[0:0].copy()
        
    return (
        clean_df.reset_index(drop=True),
        rejected_df.reset_index(drop=True),
        quality_report,
    )

def detect_day_first(series):
    """Infer DD/MM (True) or MM/DD (False) from the whole column.
    Returns None if the evidence is missing or contradictory."""
    day_first = month_first = False

    for value in series.dropna().astype(str).str.strip():
        match = re.fullmatch(r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})", value)
        if not match:
            continue
        first, second = int(match.group(1)), int(match.group(2))
        if first > 12:
            day_first = True
        if second > 12:
            month_first = True

    if day_first and not month_first:
        return True
    if month_first and not day_first:
        return False
    return None