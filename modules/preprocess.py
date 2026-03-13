"""
Text Preprocessing Module
"""

import re
import spacy
import subprocess
import sys

class TextPreprocessor:
    def __init__(self):

        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("Downloading spaCy model...")
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        print("Preprocessing model loaded successfully!")
    
    def clean_text(self, text):
        if not text or not isinstance(text, str):
            return ""
        
     
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = ' '.join(text.split())
        
        return text
    
    def remove_stopwords(self, text):
        doc = self.nlp(text)
        words = [token.text for token in doc if not token.is_stop]
        return ' '.join(words)
    
    def lemmatize(self, text):

        doc = self.nlp(text)
        words = [token.lemma_ for token in doc]
        return ' '.join(words)
    
    def process(self, text, remove_stops=True, lemmatize=True):

        #  cleaning
        cleaned = self.clean_text(text)
        if not cleaned:
            return ""
        
        # stopwords 
        if remove_stops:
            no_stops = self.remove_stopwords(cleaned)
        else:
            no_stops = cleaned
        
        #  Lemmatization
        if lemmatize:
            final = self.lemmatize(no_stops)
        else:
            final = no_stops
        
        return final
    
    def extract_keywords(self, text, num_keywords=5):
       
        keywords = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']]
 
        keywords = list(dict.fromkeys(keywords))
        return keywords[:num_keywords]
    
    def detect_language(self, text):
     
        # Check for Devanagari script (Hindi)
        hindi_pattern = re.compile(r'[\u0900-\u097F]')
        
        if hindi_pattern.search(text):
            return 'hi'  
        else:
            return 'en'  


if __name__ == "__main__":
    preprocessor = TextPreprocessor()
    
    # sample text
    test_text = "No electricity in my area for 3 days!!!"
    processed = preprocessor.process(test_text)
    print(f"Original: {test_text}")
    print(f"Processed: {processed}")