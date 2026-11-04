from langchain.tools import tool

from analysis_tools import (
    get_monthly_summary,
    detect_spending_anomalies,
    detect_zscore_anomalies,
    detect_isolation_forest_anomalies,
    fetch_transactions,
    fetch_category_trend,
)


def create_financial_tools(df):

    @tool
    def get_financial_summary(input: str = "") -> str:
        """Get total income, expenses, and net savings."""
        return get_monthly_summary(df)

    @tool
    def investigate_spending_increases() -> str:
        """Identify significant month-to-month spending increases."""
        return detect_spending_anomalies(df)

    @tool
    def investigate_unusual_transactions() -> str:
        """Detect unusual individual transactions using Z-score."""
        return detect_zscore_anomalies(df)

    @tool
    def investigate_ml_anomalies() -> str:
        """Detect unusual transactions using Isolation Forest."""
        return detect_isolation_forest_anomalies(df)

    @tool
    def get_category_trend(category: str = "") -> str:
        """Monthly expense totals for one category (empty = all expenses), with change vs previous month."""
        return fetch_category_trend(df, category or None)

    @tool
    def list_transactions(category: str = "", month: str = "") -> str:
        """List individual expense transactions, largest first (max 20). Optional filters: category name, month as YYYY-MM."""
        return fetch_transactions(df, category or None, month or None)

    return [
        get_financial_summary,
        investigate_spending_increases,
        investigate_unusual_transactions,
        investigate_ml_anomalies,
        get_category_trend,
        list_transactions,
    ]