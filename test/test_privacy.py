
from analysis_tools import load_transactions
from privacy import anonymize_transactions

df = load_transactions()

safe_df = anonymize_transactions(df)

print("Original columns:")
print(df.columns.tolist())

print("\nSafe columns:")
print(safe_df.columns.tolist())

print("\nSample sanitized data:")
print(safe_df.head().to_string(index=False))
