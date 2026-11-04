import os
from dotenv import load_dotenv
from groq import Groq

# شحن المتغيرات من ملف .env
load_dotenv()

# التحقق من وجود المفتاح في البيئة
if not os.getenv("GROQ_API_KEY"):
    print("⚠️ خطأ: لم يتم العثور على GROQ_API_KEY في ملف .env")
else:
    # إنشاء العميل (Client) باستخدام أحدث مكتبة رسمية
    # المكتبة تلتقط GROQ_API_KEY تلقائياً من الـ env دون الحاجة لتمريره يدوياً
    client = Groq()
    
    print("Available Groq Models (Text Generation):")
    print("-" * 60)
    
    try:
        # جلب قائمة النماذج الفعالة
        models_list = client.models.list()
        
        for model in models_list.data:
            # تصفية النماذج لضمان استبعاد نماذج الـ Audio أو الـ Whisper إن وُجدت
            # والتركيز على نماذج النصوص (مثل عائلات Llama و Mistral)
            if model.id and not any(x in model.id.lower() for x in ["whisper", "audio"]):
                print(f"- {model.id}")
                print(f"  Owned By: {model.owned_by}")
                # ملاحظة: Groq لا يعيد الـ Context Window مباشرة عبر الـ API الفردي للموديل،
                # ولكن يمكنك معرفتها من لوحة تحكم GroqCloud (غالباً ما تكون 8k أو 128k لنماذج Llama).
                print("-" * 60)
                
    except Exception as e:
        print(f"❌ حدث خطأ أثناء الاتصال بـ Groq: {e}")