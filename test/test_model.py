from pathlib import Path
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

model = init_chat_model(
    "openai/gpt-oss-120b",
    model_provider="groq",
    timeout=60,
    max_retries=2,
)

response = model.invoke(
    "Reply with exactly: Gemini is working."
)

print(response.content)