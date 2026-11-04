from dotenv import load_dotenv
from groq import Groq

load_dotenv()

if __name__ == "__main__":
    client = Groq()
    print("Available Models:")
    for model in client.models.list().data:
        print(f"- {model.id}")