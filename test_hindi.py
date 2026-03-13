# test_hindi.py
from modules.translator import HindiTranslator
from modules.classify import IntentClassifier
from modules.preprocess import TextPreprocessor

print("=" * 60)
print("TESTING HINDI TRANSLATION")
print("=" * 60)

# Initialize modules
translator = HindiTranslator()
classifier = IntentClassifier()
preprocessor = TextPreprocessor()

# Test Hindi phrases
test_phrases = [
    "बिजली नहीं आ रही है",
    "पानी की सप्लाई बंद है",
    "सड़क टूटी हुई है", 
    "अस्पताल में सफाई नहीं है",
    "कचरा नहीं उठा"
]

for hindi_text in test_phrases:
    print(f"\n📝 Hindi Input: {hindi_text}")
    
    # Step 1: Translate
    english_text, lang = translator.process_text(hindi_text)
    print(f"🔄 Translated: {english_text}")
    
    # Step 2: Preprocess
    processed = preprocessor.process(english_text)
    print(f"🔧 Processed: {processed}")
    
    # Step 3: Classify
    intent, confidence = classifier.predict(processed)
    print(f"🎯 Intent: {intent} ({confidence:.2f}%)")
    print("-" * 40)