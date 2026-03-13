"""
Translation Module
Handles Hindi to English translation
"""

import re
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import warnings
warnings.filterwarnings('ignore')

class HindiTranslator:
    def __init__(self):
        """
        Initialize the Hindi to English translator
        """
        print("Loading Hindi-English translation model...")
        self.model = None
        self.tokenizer = None
        self.use_fallback = True
        
        try:
            # Load mBART model for Hindi-English translation
            model_name = "facebook/mbart-large-50-many-to-many-mmt"
            self.tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
            self.model = MBartForConditionalGeneration.from_pretrained(model_name)
            
            # Set source and target languages
            self.tokenizer.src_lang = "hi_IN"
            self.use_fallback = False
            print("✅ Translation model loaded successfully!")
            
        except Exception as e:
            print(f"⚠️ Could not load translation model: {e}")
            print("⚠️ Using fallback translation system")
            self.init_fallback_translator()
    
    def init_fallback_translator(self):
        """Initialize simple fallback translator"""
        # Simple Hindi-English dictionary for common words
        self.hindi_to_english = {
            # Common complaints
            "बिजली": "electricity",
            "पानी": "water",
            "सड़क": "road",
            "अस्पताल": "hospital",
            "डॉक्टर": "doctor",
            "दवाई": "medicine",
            "कचरा": "garbage",
            "नाली": "drain",
            "लाइट": "light",
            "बल्ब": "bulb",
            "मीटर": "meter",
            "बिल": "bill",
            
            # Common verbs
            "नहीं": "no",
            "आ रही": "coming",
            "गई": "gone",
            "टूटी": "broken",
            "खराब": "bad",
            "बंद": "closed",
            "चालू": "working",
            "जल्दी": "quick",
            
            # Time related
            "कल": "yesterday",
            "आज": "today",
            "सुबह": "morning",
            "शाम": "evening",
            "रात": "night",
            "दिन": "day",
            "हफ्ता": "week",
            "महीना": "month",
            
            # Location/area
            "यहाँ": "here",
            "मोहल्ला": "area",
            "गली": "street",
            "घर": "house",
            
            # Intent keywords
            "समस्या": "problem",
            "शिकायत": "complaint",
            "मदद": "help",
            "ठीक": "fix"
        }
        
        # Common complaint patterns
        self.complaint_patterns = [
            (r'बिजली.*नहीं', "no electricity"),
            (r'पानी.*नहीं', "no water"),
            (r'सड़क.*टूटी', "broken road"),
            (r'गंदगी.*बहुत', "too dirty"),
            (r'अस्पताल.*साफ', "hospital clean"),
            (r'डॉक्टर.*नहीं', "no doctor"),
            (r'कचरा.*जमा', "garbage piled"),
        ]
    
    def detect_language(self, text):
        """
        Detect if text is Hindi or English
        
        Args:
            text: Input text
            
        Returns:
            'hi' for Hindi, 'en' for English
        """
        # Check for Devanagari script (Hindi)
        hindi_pattern = re.compile(r'[\u0900-\u097F]')
        
        if hindi_pattern.search(text):
            return 'hi'
        else:
            return 'en'
    
    def translate_hindi_to_english(self, hindi_text):
        """
        Translate Hindi text to English using mBART
        
        Args:
            hindi_text: Text in Hindi
            
        Returns:
            English translation
        """
        if not hindi_text or hindi_text.strip() == "":
            return ""
        
        if self.use_fallback:
            return self.fallback_translate(hindi_text)
        
        try:
            # Tokenize and translate
            encoded = self.tokenizer(hindi_text, return_tensors="pt", padding=True)
            generated_tokens = self.model.generate(
                **encoded,
                forced_bos_token_id=self.tokenizer.lang_code_to_id["en_XX"]
            )
            translation = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            return translation
            
        except Exception as e:
            print(f"Translation error: {e}")
            return self.fallback_translate(hindi_text)
    
    def fallback_translate(self, hindi_text):
        """
        Simple rule-based fallback translation
        """
        # Split into words
        words = hindi_text.split()
        translated_words = []
        
        for word in words:
            # Check dictionary
            if word in self.hindi_to_english:
                translated_words.append(self.hindi_to_english[word])
            else:
                # Keep original if not found
                translated_words.append(word)
        
        translated = ' '.join(translated_words)
        
        # Check patterns
        for pattern, replacement in self.complaint_patterns:
            if re.search(pattern, hindi_text):
                return replacement + " " + translated
        
        return translated
    
    def process_text(self, text):
        """
        Main function to process text - detect language and translate if needed
        
        Args:
            text: Input text (Hindi or English)
            
        Returns:
            English text, original language
        """
        if not text:
            return "", "unknown"
        
        # Detect language
        lang = self.detect_language(text)
        
        if lang == 'hi':
            # Translate Hindi to English
            print(f"🔤 Translating from Hindi: {text}")
            translated = self.translate_hindi_to_english(text)
            print(f"✅ Translated to: {translated}")
            return translated, 'hi'
        else:
            # Already English
            return text, 'en'

# For testing
if __name__ == "__main__":
    translator = HindiTranslator()
    
    # Test Hindi phrases
    test_phrases = [
        "बिजली नहीं आ रही है",
        "पानी की सप्लाई बंद है",
        "सड़क टूटी हुई है",
        "अस्पताल में सफाई नहीं है"
    ]
    
    for hindi in test_phrases:
        english, lang = translator.process_text(hindi)
        print(f"\nHindi: {hindi}")
        print(f"English: {english}")