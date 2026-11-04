import uuid

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from data_cleaner import prepare_csv, validate_and_clean
from privacy import anonymize_transactions
from tools import create_financial_tools
from agent import create_finance_agent


DEFAULT_QUESTION = (
    "Investigate the financial transactions. "
    "Summarize income and expenses, identify "
    "spending increases, detect unusual transactions "
    "using statistical and ML methods, and provide "
    "evidence-based recommendations."
)


def build_financial_agent(file_path, date_format=None):
    # Step 1: Load and prepare CSV
    raw_df = prepare_csv(file_path)

    # Step 2: Validate and clean transactions
    clean_df, rejected_df, quality_report = validate_and_clean(
        raw_df,
        date_format=date_format
    )

    # Step 3: Check data quality
    if quality_report["processing_status"] != "SUCCESS":
        return None, quality_report

    if clean_df.empty:
        return None, quality_report

    # Step 4: Apply privacy protection
    analysis_df = anonymize_transactions(
        clean_df,
        remove_label=True
    )

    # Step 5: Create financial tools
    financial_tools = create_financial_tools(analysis_df)

    # Step 6: Create agent
    agent = create_finance_agent(financial_tools)

    return agent, quality_report


def run_financial_investigation(file_path, question=None, date_format=None):
    try:
        agent, quality_report = build_financial_agent(
            file_path, date_format=date_format
        )
    except ValueError as e:
        print(f"\nData error: {e}")
        return

    print("\nData Quality Report:")
    print(quality_report)

    if agent is None:
        print("\nAnalysis stopped due to data quality issues.")
        return

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question or DEFAULT_QUESTION,
                }
            ]
        },
        config={
            "configurable": {
                "thread_id": f"finance-investigation-{uuid.uuid4()}"
            }
        },
    )
    
    total = 0
    for msg in result["messages"]:
        usage = getattr(msg, "usage_metadata", None)
        if usage:
            print(f"LLM call: {usage['input_tokens']} in / {usage['output_tokens']} out")
            total += usage["total_tokens"]
    print("Total tokens:", total)

    for msg in result["messages"]:
        for call in getattr(msg, "tool_calls", None) or []:
            print("TOOL:", call["name"], call["args"])

    print("\nFinancial Investigation Report:\n")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    file_path = Path(__file__).parent / "data" / "transactions.csv"

    run_financial_investigation(
        file_path,
        question="Investigate my finances and tell me what needs attention.",
        date_format=None,
    )