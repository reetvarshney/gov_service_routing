"""
Speech-to-Text Module with Improved Hindi Support
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
            # Use 'base' or 'small' for better Hindi accuracy
            # 'tiny' is faster but less accurate for Hindi
            self.model = whisper.load_model("base")  # Better for Hindi
            print("✅ Speech model loaded successfully! Supports Hindi and English")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
    
    def transcribe(self, audio_path):
        """
        Convert audio file to text (supports Hindi and English)
        """
        if not os.path.exists(audio_path):
            return "Error: Audio file not found"
        
        if self.model is None:
            return "Speech recognition not available"
        
        try:
            print(f"Transcribing audio file: {audio_path}")
            
            # Transcribe with language auto-detection
            result = self.model.transcribe(
                audio_path,
                language=None,  # Auto-detect language
                task="transcribe",
                fp16=False,  # Use float32 (compatible with CPU)
                temperature=0.0,  # Lower temperature for more accurate transcription
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6,
                condition_on_previous_text=False  # Better for code-switching
            )
            
            text = result["text"].strip()
            detected_language = result.get("language", "unknown")
            print(f"📢 Detected language: {detected_language}")
            print(f"📝 Transcribed text: {text}")
            
            if text:
                return text
            else:
                return "Could not understand audio. Please try again."
                
        except Exception as e:
            print(f"Error in transcription: {e}")
            return f"Error in transcription: {str(e)}"
    
    def transcribe_microphone(self, audio_data):
        """
        Transcribe from microphone input
        """
        if self.model is None:
            return "Speech recognition not available"
        
        try:
            # Handle different audio formats
            if isinstance(audio_data, tuple) and len(audio_data) == 2:
                sample_rate, audio_array = audio_data
                
                print(f"Processing microphone audio: {sample_rate}Hz")
                
                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                    import scipy.io.wavfile as wav
                    wav.write(tmp.name, sample_rate, audio_array.astype(np.int16))
                    tmp_path = tmp.name
                
                # Transcribe
                text = self.transcribe(tmp_path)
                
                # Clean up
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                
                return text
            else:
                print(f"Unexpected audio format: {type(audio_data)}")
                return "Invalid audio format"
                
        except Exception as e:
            print(f"Error processing microphone: {e}")
            return f"Error processing microphone: {str(e)}"