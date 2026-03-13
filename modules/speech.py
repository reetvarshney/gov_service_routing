"""
Speech-to-Text Module with Hindi Support
"""

import whisper
import os
import tempfile
import numpy as np

class SpeechProcessor:
    def __init__(self):
        """
        Initialize the speech processor with multilingual Whisper model
        """
        print("Loading multilingual speech recognition model...")
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load Whisper model with multilingual support"""
        try:
            # 'base' model works well for Hindi, 'tiny' is faster but less accurate
            self.model = whisper.load_model("base")  # Can also use "tiny" for speed
            print("✅ Speech model loaded successfully! Supports Hindi and English")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            print("⚠️ Will attempt to use fallback")
    
    def transcribe(self, audio_path):
        """
        Convert audio file to text (supports Hindi and English)
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text (in original language)
        """
        if not os.path.exists(audio_path):
            return "Error: Audio file not found"
        
        if self.model is None:
            return "Speech recognition not available"
        
        try:
            # Transcribe with language auto-detection
            result = self.model.transcribe(
                audio_path,
                language=None,  # Auto-detect language
                task="transcribe",  # Just transcribe, don't translate
                fp16=False  # Use float32 (compatible with CPU)
            )
            
            text = result["text"].strip()
            detected_language = result.get("language", "unknown")
            print(f"📢 Detected language: {detected_language}")
            
            if text:
                return text
            else:
                return "Could not understand audio. Please try again."
                
        except Exception as e:
            return f"Error in transcription: {str(e)}"
    
    def transcribe_microphone(self, audio_data):
        """
        Transcribe from microphone input
        
        Args:
            audio_data: Audio data from microphone (sample_rate, audio_array)
            
        Returns:
            Transcribed text (in original language)
        """
        if self.model is None:
            return "Speech recognition not available"
        
        try:
            # Handle different audio formats
            if isinstance(audio_data, tuple) and len(audio_data) == 2:
                sample_rate, audio_array = audio_data
                
                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                    import scipy.io.wavfile as wav
                    wav.write(tmp.name, sample_rate, audio_array.astype(np.int16))
                    tmp_path = tmp.name
                
                # Transcribe
                text = self.transcribe(tmp_path)
                
                # Clean up
                os.unlink(tmp_path)
                
                return text
            else:
                return "Invalid audio format"
                
        except Exception as e:
            return f"Error processing microphone: {str(e)}"
    
    def transcribe_and_translate(self, audio_path, target_language="en"):
        """
        Transcribe and optionally translate to target language
        
        Args:
            audio_path: Path to audio file
            target_language: Target language code (e.g., 'en' for English)
            
        Returns:
            Transcribed and translated text
        """
        if self.model is None:
            return "Speech recognition not available"
        
        try:
            # Transcribe with translation task
            result = self.model.transcribe(
                audio_path,
                task="translate",  # Translate to English
                fp16=False
            )
            
            text = result["text"].strip()
            detected_language = result.get("language", "unknown")
            print(f"📢 Detected language: {detected_language}")
            print(f"✅ Translated to: {text}")
            
            return text
            
        except Exception as e:
            return f"Error: {str(e)}"