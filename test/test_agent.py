from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from main import build_financial_agent

# Build the agent using the test dataset
file_path = Path(__file__).parent.parent / "data" / "test_transactions.csv"

agent, quality_report = build_financial_agent(
    file_path,
    date_format="%m/%d/%Y"
)

config = {
    "configurable": {
        "thread_id": "finance-test-001"
    }
}

# Test 1: Conversation memory
result1 = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "The test file name is test_finance.csv. Please acknowledge."
            }
        ]
    },
    config=config
)

print("First response:")
print(result1["messages"][-1].content)

# Test 2: Recall previous information
result2 = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is the name of the test file I mentioned?"
            }
        ]
    },
    config=config
)

print("\nSecond response:")
print(result2["messages"][-1].content)

def test_tool_selection(question, thread_id):
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ]
        },
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    print("\nExecution trace:")

    for message in result["messages"]:
        print(f"\nMessage type: {message.type}")

        if getattr(message, "name", None):
            print(f"Name: {message.name}")

        for call in getattr(message, "tool_calls", []):
            print(f"Tool call: {call['name']}")
            print(f"Arguments: {call['args']}")

        if message.type == "ai" and message.content:
            print("\nAgent Response:")
            print(message.content)

# Comprehensive investigation test
test_tool_selection(
    "Conduct a comprehensive financial investigation. "
    "Summarize financial performance, identify spending increases, "
    "detect statistical and ML anomalies, compare the findings, "
    "and provide evidence-based recommendations.",
    "tool-test-009"
)