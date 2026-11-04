import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

# المكتبة تلتقط GOOGLE_API_KEY تلقائياً من الـ env
client = genai.Client()

print("Available Gemini Models (Text Generation):")
print("-" * 60)

for model in client.models.list():
    # تعديل: استخدام supported_actions بدلاً من supported_methods
    # وتغيير generateContent إلى الصيغة الجديدة العادية أو التحقق من وجودها
    if model.supported_actions and any("generateContent" in action for action in model.supported_actions):
        # الاسم يأتي جاهزاً ونظيفاً في المكتبة الجديدة بدون بادئة models/
        print(f"- {model.name}")
        print(f"  Max Input Tokens : {model.input_token_limit}")
        print(f"  Max Output Tokens: {model.output_token_limit}")
        print("-" * 60)