"""
Text Preprocessing Module
Cleans and prepares text for classification
"""

import re
import spacy
import subprocess
import sys

class TextPreprocessor:
    def __init__(self):
        """
        Initialize the preprocessor with spaCy model
        """
        print("Loading text preprocessing model...")
        
        # Try to load spaCy model, download if not available
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("Downloading spaCy model...")
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        print("Preprocessing model loaded successfully!")
    
    def clean_text(self, text):
        """
        Basic text cleaning
        
        Args:
            text: Raw text input
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits (keep only letters and spaces)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra spaces
        text = ' '.join(text.split())
        
        return text
    
    def remove_stopwords(self, text):
        """
        Remove common stopwords
        
        Args:
            text: Cleaned text
            
        Returns:
            Text with stopwords removed
        """
        doc = self.nlp(text)
        words = [token.text for token in doc if not token.is_stop]
        return ' '.join(words)
    
    def lemmatize(self, text):
        """
        Convert words to base form
        
        Args:
            text: Text to lemmatize
            
        Returns:
            Lemmatized text
        """
        doc = self.nlp(text)
        words = [token.lemma_ for token in doc]
        return ' '.join(words)
    
    def process(self, text, remove_stops=True, lemmatize=True):
        """
        Full preprocessing pipeline
        
        Args:
            text: Raw input text
            remove_stops: Whether to remove stopwords
            lemmatize: Whether to lemmatize words
            
        Returns:
            Processed text ready for classification
        """
        # Step 1: Basic cleaning
        cleaned = self.clean_text(text)
        
        if not cleaned:
            return ""
        
        # Step 2: Remove stopwords (optional)
        if remove_stops:
            no_stops = self.remove_stopwords(cleaned)
        else:
            no_stops = cleaned
        
        # Step 3: Lemmatize (optional)
        if lemmatize:
            final = self.lemmatize(no_stops)
        else:
            final = no_stops
        
        return final
    
    def extract_keywords(self, text, num_keywords=5):
        """
        Extract important keywords from text
        
        Args:
            text: Input text
            num_keywords: Number of keywords to extract
            
        Returns:
            List of keywords
        """
        doc = self.nlp(text)
        
        # Get nouns and proper nouns
        keywords = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']]
        
        # Remove duplicates and return top N
        keywords = list(dict.fromkeys(keywords))
        return keywords[:num_keywords]
    
    def detect_language(self, text):
        """
        Simple language detection (basic)
        
        Args:
            text: Input text
            
        Returns:
            'hi' for Hindi/English mixed, 'en' for English
        """
        # Check for Devanagari script (Hindi)
        hindi_pattern = re.compile(r'[\u0900-\u097F]')
        
        if hindi_pattern.search(text):
            return 'hi'  # Contains Hindi
        else:
            return 'en'  # English only

# For testing
if __name__ == "__main__":
    preprocessor = TextPreprocessor()
    
    # Test with sample text
    test_text = "No electricity in my area for 3 days!!!"
    processed = preprocessor.process(test_text)
    print(f"Original: {test_text}")
    print(f"Processed: {processed}")