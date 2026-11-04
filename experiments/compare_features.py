from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score

from analysis_tools import add_rolling_features, CONTAMINATION


def load_evaluation_transactions():
    """Load the labeled synthetic dataset for offline ML evaluation only."""
    file_path = Path(__file__).parent.parent / "data" / "transactions.csv"
    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"], errors="raise")

    if "label" not in df.columns:
        raise ValueError("The evaluation dataset must contain a label column.")

    return df


def compare_isolation_features():
    """Compare Isolation Forest with and without rolling time features."""
    df = load_evaluation_transactions()
    expenses = df[df["type"] == "expense"].copy()
    expenses = add_rolling_features(expenses)

    y_true = expenses["label"].astype(int)

    feature_sets = {
        "Basic (Amount + Category)": ["amount", "category"],
        "With Rolling Features": [
            "amount", "category", "rolling_count", "rolling_sum"
        ],
    }

    for name, columns in feature_sets.items():
        features = pd.get_dummies(
            expenses[columns], columns=["category"], dtype=int
        )

        model = IsolationForest(
            n_estimators=200,
            contamination=CONTAMINATION,
            random_state=42,
        )
        y_pred = (model.fit_predict(features) == -1).astype(int)

        print(f"\n{'=' * 40}\n{name}\n{'=' * 40}")
        print("Detected:", int(y_pred.sum()))
        print("Precision:", round(precision_score(y_true, y_pred, zero_division=0) * 100, 2), "%")
        print("Recall:", round(recall_score(y_true, y_pred, zero_division=0) * 100, 2), "%")
        print("F1-score:", round(f1_score(y_true, y_pred, zero_division=0) * 100, 2), "%")

        print("\nDetected transactions:")
        print(expenses[y_pred == 1][["date", "category", "amount", "label"]].to_string(index=False))


if __name__ == "__main__":
    compare_isolation_features()