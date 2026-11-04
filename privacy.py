
import pandas as pd
import re


def anonymize_transactions(df, remove_label=False):
    """
    Remove or mask sensitive personal information
    before sending transactions to the AI Agent.
    """

    df = df.copy()

    # Remove direct personal identifiers
    sensitive_columns = {
        "name",
        "full_name",
        "email",
        "phone",
        "account_number",
        "iban",
        "address"
    }
    
    if remove_label:
        sensitive_columns.add("label")
        
    df = df.drop(
        columns=[
            col for col in df.columns
            if col.strip().lower() in sensitive_columns
        ],
        errors="ignore"
    )

    # Mask possible emails and phone numbers in descriptions
    description_col = next(
        (col for col in df.columns if col.strip().lower() == "description"),
        None
    )
    
    if description_col:

        def mask_sensitive_text(value):
            if pd.isna(value):
                return value

            value = str(value)

            # Mask email addresses
            value = re.sub(
                r'[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}',
                '[EMAIL]',
                value
            )

            # Mask phone-like numbers
            value = re.sub(
                r'\+?\d[\d\s\-()]{7,}\d',
                '[PHONE]',
                value
            )

            return value

        df[description_col] = df[description_col].apply(
            mask_sensitive_text
        )

    return df
