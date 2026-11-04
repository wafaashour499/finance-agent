import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest

# ---------------- Settings ----------------
ZSCORE_THRESHOLD = 2.0
MIN_TRANSACTIONS = 5
SPIKE_PCT = 30
ROLLING_DAYS = 5
CONTAMINATION = 0.02
MAX_RESULTS = 10


def load_transactions(df=None):
    """Prepare cleaned transactions for financial analysis."""

    if df is None:
        file_path = Path(__file__).parent / "data" / "transactions.csv"
        df = pd.read_csv(file_path)

    df = df.copy()

    # Convert the Data Cleaner schema to the internal analysis schema.
    df = df.rename(columns={
        "Date": "date",
        "Amount": "amount",
        "Category": "category",
        "Description": "description",
        "Type": "type",
    })

    # Labels are only for offline model evaluation.
    df = df.drop(columns=["label"], errors="ignore")

    required_columns = ["date", "amount", "category", "description", "type"]
    missing_columns = [c for c in required_columns if c not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing analysis columns: {missing_columns}")

    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df["amount"] = pd.to_numeric(df["amount"], errors="raise")

    return df


def add_rolling_features(expenses):
    """Add rolling count/sum of transactions per category (time window)."""
    expenses = expenses.sort_values("date").reset_index(drop=True)
    expenses["rolling_count"] = 0
    expenses["rolling_sum"] = 0.0

    for category in expenses["category"].unique():
        mask = expenses["category"] == category
        category_data = expenses.loc[mask]

        counts = []
        sums = []

        for date in category_data["date"]:
            window = category_data[
                (category_data["date"] <= date) &
                (category_data["date"] > date - pd.Timedelta(days=ROLLING_DAYS))
            ]
            counts.append(len(window))
            sums.append(float(window["amount"].sum()))

        expenses.loc[mask, "rolling_count"] = counts
        expenses.loc[mask, "rolling_sum"] = sums

    return expenses


def get_monthly_summary(df=None):
    """Calculate total income, expenses, and net savings."""
    df = load_transactions(df)

    income = df[df["type"] == "income"]["amount"].sum()
    expenses = df[df["type"] == "expense"]["amount"].sum()

    return json.dumps({
        "total_income": round(float(income), 2),
        "total_expenses": round(float(expenses), 2),
        "net_savings": round(float(income - expenses), 2),
    }, ensure_ascii=False)


def detect_spending_anomalies(df=None):
    """Detect significant spending increases between consecutive months."""
    df = load_transactions(df)

    expenses = df[df["type"] == "expense"].copy()
    expenses["month"] = expenses["date"].dt.to_period("M").astype(str)

    monthly = expenses.pivot_table(
        index="month",
        columns="category",
        values="amount",
        aggfunc="sum",
        fill_value=0,
    ).sort_index()

    results = []

    for i in range(1, len(monthly)):
        previous = monthly.iloc[i - 1]
        current = monthly.iloc[i]

        for category in monthly.columns:
            old_amount = float(previous[category])
            new_amount = float(current[category])

            if new_amount <= old_amount:
                continue

            if old_amount == 0:
                results.append({
                    "category": category,
                    "previous_month": monthly.index[i - 1],
                    "current_month": monthly.index[i],
                    "previous_amount": 0,
                    "current_amount": new_amount,
                    "increase_pct": None,
                    "status": "new_category",
                })
            else:
                increase_pct = ((new_amount - old_amount) / old_amount) * 100

                if increase_pct >= SPIKE_PCT:
                    results.append({
                        "category": category,
                        "previous_month": monthly.index[i - 1],
                        "current_month": monthly.index[i],
                        "previous_amount": old_amount,
                        "current_amount": new_amount,
                        "increase_pct": round(increase_pct, 2),
                        "status": "spending_increase",
                    })

    return json.dumps(results, ensure_ascii=False)


def detect_zscore_anomalies(
    df=None,
    threshold=ZSCORE_THRESHOLD,
    min_transactions=MIN_TRANSACTIONS,
    max_results=MAX_RESULTS,
):
    """Detect unusually large transactions within each spending category."""
    df = load_transactions(df)

    expenses = df[df["type"] == "expense"].copy()
    results = []

    for category, group in expenses.groupby("category"):

        if len(group) < min_transactions:
            continue

        mean = group["amount"].mean()
        std = group["amount"].std(ddof=0)

        if std == 0:
            continue

        group = group.assign(z_score=(group["amount"] - mean) / std)
        anomalies = group[group["z_score"] >= threshold]

        for _, row in anomalies.iterrows():
            results.append({
                "category": category,
                "date": row["date"].strftime("%Y-%m-%d"),
                "description": row["description"],
                "amount": float(row["amount"]),
                "z_score": round(float(row["z_score"]), 2),
            })

    results = sorted(results, key=lambda x: x["z_score"], reverse=True)
    return json.dumps(results[:max_results], ensure_ascii=False)


def detect_iqr_anomalies(df=None, min_transactions=MIN_TRANSACTIONS):
    """Detect unusual individual transactions using the IQR method."""
    df = load_transactions(df)
    expenses = df[df["type"] == "expense"].copy()

    results = []

    for category, group in expenses.groupby("category"):

        if len(group) < min_transactions:
            continue

        q1 = group["amount"].quantile(0.25)
        q3 = group["amount"].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

        anomalies = group[
            (group["amount"] < lower_bound) |
            (group["amount"] > upper_bound)
        ]

        for _, row in anomalies.iterrows():
            results.append({
                "date": row["date"].strftime("%Y-%m-%d"),
                "category": category,
                "description": row["description"],
                "amount": float(row["amount"]),
                "lower_bound": round(float(lower_bound), 2),
                "upper_bound": round(float(upper_bound), 2),
            })

    return json.dumps(results, ensure_ascii=False)


def detect_isolation_forest_anomalies(df=None):
    """Detect point and collective anomalies using Isolation Forest."""
    df = load_transactions(df)

    expenses = df[df["type"] == "expense"].copy()
    expenses = add_rolling_features(expenses)

    features = pd.get_dummies(
        expenses[["amount", "category", "rolling_count", "rolling_sum"]],
        columns=["category"],
        dtype=int,
    )

    model = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        random_state=42,
    )

    expenses["prediction"] = model.fit_predict(features)
    anomalies = expenses[expenses["prediction"] == -1]

    results = []

    for _, row in anomalies.iterrows():
        results.append({
            "date": row["date"].strftime("%Y-%m-%d"),
            "description": row["description"],
            "category": row["category"],
            "amount": float(row["amount"]),
            "rolling_count": int(row["rolling_count"]),
            "rolling_sum": float(row["rolling_sum"]),
        })

    return json.dumps(results, ensure_ascii=False)

def fetch_transactions(df=None, category=None, month=None, limit=20):
    """List expense transactions (largest first). Optional filters: category, month (YYYY-MM)."""
    df = load_transactions(df)
    data = df[df["type"] == "expense"].copy()

    if category:
        data = data[data["category"].str.casefold() == category.strip().casefold()]
    if month:
        data = data[data["date"].dt.strftime("%Y-%m") == month.strip()]

    data = data.sort_values("amount", ascending=False)
    rows = [
        {
            "date": row["date"].strftime("%Y-%m-%d"),
            "category": row["category"],
            "description": row["description"],
            "amount": float(row["amount"]),
        }
        for _, row in data.head(limit).iterrows()
    ]

    return json.dumps(
        {"total_matches": len(data), "returned": len(rows), "transactions": rows},
        ensure_ascii=False,
    )


def fetch_category_trend(df=None, category=None):
    """Monthly expense totals for one category (or all expenses if no category)."""
    df = load_transactions(df)
    expenses = df[df["type"] == "expense"].copy()

    if category:
        expenses = expenses[
            expenses["category"].str.casefold() == category.strip().casefold()
        ]
        if expenses.empty:
            available = sorted(
                df[df["type"] == "expense"]["category"].unique().tolist()
            )
            return json.dumps(
                {"error": f"Category '{category}' not found.", "available_categories": available},
                ensure_ascii=False,
            )

    expenses["month"] = expenses["date"].dt.to_period("M").astype(str)
    monthly = expenses.groupby("month")["amount"].agg(["sum", "count", "mean"])

    results = []
    previous_total = None

    for month, row in monthly.iterrows():
        total = float(row["sum"])
        change_pct = None
        if previous_total:
            change_pct = round((total - previous_total) / previous_total * 100, 2)

        results.append({
            "month": month,
            "total": round(total, 2),
            "transactions": int(row["count"]),
            "average": round(float(row["mean"]), 2),
            "change_vs_previous_pct": change_pct,
        })
        previous_total = total

    return json.dumps(
        {"category": category or "all expenses", "months": results},
        ensure_ascii=False,
    )

if __name__ == "__main__":
    from data_cleaner import prepare_csv, validate_and_clean

    file_path = Path(__file__).parent / "data" / "transactions.csv"

    raw_df = prepare_csv(file_path)
    clean_df, rejected_df, report = validate_and_clean(
        raw_df, date_format="%m/%d/%Y"
    )

    if report["processing_status"] == "SUCCESS":
        print("\nMonthly Summary:")
        print(get_monthly_summary(clean_df))

        print("\nSpending Increases:")
        print(detect_spending_anomalies(clean_df))

        print("\nZ-Score Anomalies:")
        print(detect_zscore_anomalies(clean_df))

        print("\nIsolation Forest Anomalies:")
        print(detect_isolation_forest_anomalies(clean_df))
    else:
        print(report)