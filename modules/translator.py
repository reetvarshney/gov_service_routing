"""
Translation Module
Handles Hindi to English translation with improved accuracy
"""

import re
import subprocess
import sys

class HindiTranslator:
    def __init__(self):
        """
        Initialize the Hindi to English translator
        """
        print("Loading Hindi-English translation system...")
        self.use_fallback = True
        self.model = None
        self.tokenizer = None
        
        # Try to load better translation model
        try:
            from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
            import torch
            
            print("Loading mBART model for Hindi-English translation...")
            model_name = "facebook/mbart-large-50-many-to-many-mmt"
            self.tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
            self.model = MBartForConditionalGeneration.from_pretrained(model_name)
            self.tokenizer.src_lang = "hi_IN"
            self.use_fallback = False
            print("✅ Advanced translation model loaded!")
            
        except Exception as e:
            print(f"⚠️ Could not load advanced model: {e}")
            print("⚠️ Using enhanced fallback translator")
            self.init_enhanced_fallback()
    
    def init_enhanced_fallback(self):
        """Initialize enhanced fallback translator with more words and patterns"""
        
        # Extensive Hindi-English dictionary
        self.hindi_to_english = {
            # Electricity related
            "बिजली": "electricity",
            "लाइट": "light",
            "बल्ब": "bulb",
            "मीटर": "meter",
            "तार": "wire",
            "पोल": "pole",
            "ट्रांसफार्मर": "transformer",
            "वोल्टेज": "voltage",
            "करंट": "current",
            "फ्यूज": "fuse",
            "बिजली बिल": "electricity bill",
            "बिजली कटौती": "power cut",
            
            # Water related
            "पानी": "water",
            "नल": "tap",
            "मोटर": "motor",
            "पाइप": "pipe",
            "टंकी": "tank",
            "सप्लाई": "supply",
            "पानी की समस्या": "water problem",
            "पानी निकासी": "water drainage",
            "सीवर": "sewer",
            "नाली": "drain",
            
            # Road related
            "सड़क": "road",
            "गड्ढा": "pothole",
            "पुल": "bridge",
            "फुटपाथ": "footpath",
            "स्ट्रीट लाइट": "street light",
            "ट्रैफिक": "traffic",
            "सिग्नल": "signal",
            
            # Healthcare related
            "अस्पताल": "hospital",
            "डॉक्टर": "doctor",
            "दवा": "medicine",
            "मरीज": "patient",
            "एम्बुलेंस": "ambulance",
            "इलाज": "treatment",
            "सफाई": "cleanliness",
            "स्वास्थ्य": "health",
            
            # Sanitation related
            "कचरा": "garbage",
            "गंदगी": "dirt",
            "सफाई कर्मचारी": "cleaner",
            "कूड़ादान": "dustbin",
            "झाड़ू": "broom",
            
            # Common verbs and modifiers
            "नहीं": "no",
            "आ रही": "coming",
            "गई": "gone",
            "टूटी": "broken",
            "खराब": "bad",
            "बंद": "closed",
            "चालू": "working",
            "साफ": "clean",
            "गंदा": "dirty",
            "भरा": "full",
            "खाली": "empty",
            "ठीक": "fine",
            "जल्दी": "quick",
            "धीमा": "slow",
            "बहुत": "very",
            "कम": "less",
            "ज्यादा": "more",
            
            # Time related
            "कल": "yesterday",
            "आज": "today",
            "कल सुबह": "yesterday morning",
            "आज सुबह": "this morning",
            "तीन दिन": "three days",
            "एक हफ्ता": "one week",
            "एक महीना": "one month",
            
            # Location words
            "यहाँ": "here",
            "वहाँ": "there",
            "मोहल्ला": "neighborhood",
            "गली": "street",
            "घर": "house",
            "इलाका": "area",
            "शहर": "city",
            "गांव": "village",
            
            # Complaint related
            "समस्या": "problem",
            "शिकायत": "complaint",
            "मदद": "help",
            "सुविधा": "facility",
            "सेवा": "service",
        }
        
        # Common complaint patterns (regex patterns)
        self.complaint_patterns = [
            # Electricity patterns
            (r'बिजली.*नहीं.*आ', "no electricity"),
            (r'बिजली.*कट', "power cut"),
            (r'लाइट.*नहीं.*जल', "lights not working"),
            (r'बिजली.*बिल.*समस्या', "electricity bill problem"),
            (r'मीटर.*खराब', "meter not working"),
            (r'ट्रांसफार्मर.*फट', "transformer blast"),
            (r'वोल्टेज.*उतार.*चढ़ाव', "voltage fluctuation"),
            
            # Water patterns
            (r'पानी.*नहीं.*आ', "no water supply"),
            (r'पानी.*सप्लाई.*बंद', "water supply stopped"),
            (r'नल.*टूट', "tap broken"),
            (r'पाइप.*फट', "pipe burst"),
            (r'पानी.*गंदा', "dirty water"),
            (r'पानी.*लीकेज', "water leakage"),
            (r'पानी.*दबाव.*कम', "low water pressure"),
            
            # Road patterns
            (r'सड़क.*टूट', "broken road"),
            (r'सड़क.*गड्ढा', "potholes in road"),
            (r'सड़क.*खराब', "bad road condition"),
            (r'स्ट्रीट.*लाइट.*नहीं', "street light not working"),
            
            # Healthcare patterns
            (r'अस्पताल.*सफाई.*नहीं', "hospital not clean"),
            (r'डॉक्टर.*नहीं.*मिल', "doctor not available"),
            (r'दवा.*नहीं.*मिल', "medicine not available"),
            (r'एम्बुलेंस.*नहीं.*आ', "ambulance not coming"),
            
            # Sanitation patterns
            (r'कचरा.*नहीं.*उठ', "garbage not collected"),
            (r'गंदगी.*फैल', "dirt spread"),
            (r'नाली.*जाम', "drain blocked"),
            (r'सीवर.*ओवरफ्लो', "sewer overflow"),
        ]
    
    def detect_language(self, text):
        """
        Detect if text is Hindi or English
        
        Args:
            text: Input text
            
        Returns:
            'hi' for Hindi, 'en' for English
        """
        if not text:
            return 'en'
            
        # Check for Devanagari script (Hindi)
        hindi_pattern = re.compile(r'[\u0900-\u097F]')
        
        if hindi_pattern.search(text):
            return 'hi'
        else:
            return 'en'
    
    def translate_hindi_to_english(self, hindi_text):
        """
        Translate Hindi text to English
        
        Args:
            hindi_text: Text in Hindi
            
        Returns:
            English translation
        """
        if not hindi_text or hindi_text.strip() == "":
            return ""
        
        print(f"Translating: {hindi_text}")
        
        # Try advanced model first
        if not self.use_fallback:
            try:
                import torch
                inputs = self.tokenizer(hindi_text, return_tensors="pt", padding=True, truncation=True)
                translated_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tokenizer.lang_code_to_id["en_XX"],
                    max_length=128
                )
                translation = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                print(f"Model translation: {translation}")
                return translation
            except Exception as e:
                print(f"Model translation failed: {e}")
                # Fall back to enhanced fallback
        
        # Enhanced fallback translation
        return self.enhanced_fallback_translate(hindi_text)
    
    def enhanced_fallback_translate(self, hindi_text):
        """
        Enhanced rule-based translation
        """
        # First check if it matches any known pattern
        for pattern, translation in self.complaint_patterns:
            if re.search(pattern, hindi_text):
                return translation
        
        # If no pattern matches, do word-by-word translation
        words = hindi_text.split()
        translated_words = []
        
        for word in words:
            # Check dictionary
            if word in self.hindi_to_english:
                translated_words.append(self.hindi_to_english[word])
            else:
                # Try to find partial matches (for compound words)
                found = False
                for hindi_word, english_word in self.hindi_to_english.items():
                    if hindi_word in word:
                        translated_words.append(english_word)
                        found = True
                        break
                if not found:
                    # Keep original if not found (transliterate approximately)
                    translated_words.append(word)
        
        # Join translated words
        translation = ' '.join(translated_words)
        
        # Common phrase corrections
        corrections = {
            "no electricity": "no electricity",
            "electricity no coming": "no electricity",
            "water no coming": "no water supply",
            "road broken": "broken road",
            "hospital cleanliness no": "hospital not clean",
            "garbage no collected": "garbage not collected",
        }
        
        for wrong, correct in corrections.items():
            if wrong in translation.lower():
                return correct
        
        return translation
    
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
        print(f"Detected language: {lang}")
        
        if lang == 'hi':
            # Translate Hindi to English
            translated = self.translate_hindi_to_english(text)
            print(f"✅ Translated to: {translated}")
            return translated, 'hi'
        else:
            # Already English
            return text, 'en'