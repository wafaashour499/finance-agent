from pathlib import Path
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

model = init_chat_model(
    "openai/gpt-oss-20b",
    model_provider="groq",
    timeout=60,
    max_retries=2,
)

@tool
def get_test_balance() -> str:
    """Return a test financial balance."""
    return "The test balance is 1500 EGP."

model_with_tools = model.bind_tools([get_test_balance])

response = model_with_tools.invoke(
    "Use get_test_balance to retrieve the test balance."
)

print("Tool calls:")
print(response.tool_calls)