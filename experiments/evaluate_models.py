import json
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from analysis_tools import (
    detect_zscore_anomalies,
    detect_iqr_anomalies,
    detect_isolation_forest_anomalies,
)
from experiments.compare_features import load_evaluation_transactions


def transaction_key(row):
    """Create a key to match detected transactions with the dataset."""
    return (
        pd.to_datetime(row["date"]).strftime("%Y-%m-%d"),
        str(row["category"]),
        str(row["description"]),
        round(float(row["amount"]), 2),
    )


def evaluate_model(name, detector, df):
    """Evaluate one anomaly detector against the true labels."""

    results = json.loads(detector(df))

    y_true = df["label"].astype(int).tolist()
    y_pred = [0] * len(df)

    actual_keys = {
        transaction_key(row): i
        for i, row in df.iterrows()
    }

    for item in results:
        key = transaction_key(item)
        if key in actual_keys:
            y_pred[actual_keys[key]] = 1

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    print(f"\n{'=' * 35}")
    print(name)
    print("=" * 35)
    print(f"TP: {tp} | FP: {fp}")
    print(f"FN: {fn} | TN: {tn}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.2%}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.2%}")
    print(f"F1-score:  {f1_score(y_true, y_pred, zero_division=0):.2%}")


if __name__ == "__main__":
    df = load_evaluation_transactions()

    # Evaluate expenses only
    df = df[df["type"] == "expense"].copy().reset_index(drop=True)

    models = [
        ("Z-Score", lambda d: detect_zscore_anomalies(d, max_results=None)),
        ("IQR", detect_iqr_anomalies),
        ("Isolation Forest", detect_isolation_forest_anomalies),
    ]

    for name, detector in models:
        evaluate_model(name, detector, df)