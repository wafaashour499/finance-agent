# استيراد الدالة المراد اختبارها من ملف الأدوات
from analysis_tools import detect_zscore_anomalies

# 1. استدعاء الدالة وتخزين نتيجتها في متغير
test_result = detect_zscore_anomalies()

# 2. طباعة النتيجة الفعلية لرؤية محتواها (هل هي [] أم بيانات فعلية؟)
print("Output:", test_result)

# 3. طباعة نوع البيانات (Data Type) للتحقق مما إذا كانت قائمة (list)
print("Type:", type(test_result))