"""
Intent Classification Module
Classifies complaints into service categories
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

class IntentClassifier:
    def __init__(self):
        """
        Initialize the classifier
        """
        self.vectorizer = TfidfVectorizer(
            max_features=500,  # Use top 500 words
            ngram_range=(1, 2),  # Use single words and pairs
            stop_words='english'  # Remove English stopwords
        )
        self.classifier = LogisticRegression(
            max_iter=1000,
            random_state=42
        )
        self.is_trained = False
        self.model_path = "models/intent_classifier.pkl"
        
        # Try to load existing model
        self.load_model()
    
    def train(self, csv_path="data/complaints.csv"):
        """
        Train the classifier on complaint data
        
        Args:
            csv_path: Path to complaints CSV file
        """
        print(f"Loading training data from {csv_path}...")
        
        # Check if file exists
        if not os.path.exists(csv_path):
            print(f"Error: Training file not found at {csv_path}")
            return False
        
        # Load data
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} training examples")
        
        # Check if data is valid
        if len(df) == 0:
            print("Error: No training data found")
            return False
        
        # Prepare features and labels
        X = df['complaint_text']
        y = df['intent']
        
        # Convert text to vectors
        print("Converting text to features...")
        X_vectors = self.vectorizer.fit_transform(X)
        
        # Train classifier
        print("Training classifier...")
        self.classifier.fit(X_vectors, y)
        
        # Calculate accuracy (optional)
        X_train, X_test, y_train, y_test = train_test_split(
            X_vectors, y, test_size=0.2, random_state=42
        )
        self.classifier.fit(X_train, y_train)
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy:.2f}")
        
        # Retrain on full data
        self.classifier.fit(X_vectors, y)
        
        self.is_trained = True
        
        # Save the model
        self.save_model()
        
        print("Training complete!")
        return True
    
    def predict(self, text):
        """
        Predict intent for a single text
        
        Args:
            text: Preprocessed text
            
        Returns:
            intent: Predicted category
            confidence: Confidence score (0-1)
        """
        if not self.is_trained:
            print("Model not trained. Training now...")
            self.train()
        
        # Convert text to vector
        X = self.vectorizer.transform([text])
        
        # Predict
        intent = self.classifier.predict(X)[0]
        
        # Get confidence
        probabilities = self.classifier.predict_proba(X)[0]
        confidence = max(probabilities)
        
        return intent, confidence
    
    def predict_batch(self, texts):
        """
        Predict intents for multiple texts
        
        Args:
            texts: List of text strings
            
        Returns:
            List of (intent, confidence) tuples
        """
        if not self.is_trained:
            self.train()
        
        X = self.vectorizer.transform(texts)
        intents = self.classifier.predict(X)
        probabilities = self.classifier.predict_proba(X)
        
        results = []
        for i, intent in enumerate(intents):
            confidence = max(probabilities[i])
            results.append((intent, confidence))
        
        return results
    
    def save_model(self, filepath=None):
        """
        Save trained model to disk
        
        Args:
            filepath: Path to save model (optional)
        """
        if filepath is None:
            filepath = self.model_path
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save vectorizer and classifier
        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath=None):
        """
        Load trained model from disk
        
        Args:
            filepath: Path to model file (optional)
        """
        if filepath is None:
            filepath = self.model_path
        
        if os.path.exists(filepath):
            try:
                model_data = joblib.load(filepath)
                self.vectorizer = model_data['vectorizer']
                self.classifier = model_data['classifier']
                self.is_trained = True
                print(f"Model loaded from {filepath}")
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
        else:
            print(f"No existing model found at {filepath}")
            return False
    
    def get_all_intents(self):
        """
        Get list of all possible intents
        
        Returns:
            List of intent categories
        """
        if self.is_trained:
            return self.classifier.classes_.tolist()
        else:
            return []

# For testing
if __name__ == "__main__":
    classifier = IntentClassifier()
    
    # Train the model
    classifier.train()
    
    # Test prediction
    test_text = "no electricity in my house"
    intent, confidence = classifier.predict(test_text)
    print(f"Text: {test_text}")
    print(f"Intent: {intent}")
    print(f"Confidence: {confidence:.2f}")