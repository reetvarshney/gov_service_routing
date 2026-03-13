"""
Main Application - Government Service Routing System
Professional Interface with Hindi/English Support
"""

import gradio as gr
import os
import tempfile
import numpy as np
import socket
from modules.speech import SpeechProcessor
from modules.preprocess import TextPreprocessor
from modules.classify import IntentClassifier
from modules.routing import PortalRouter
from modules.translator import HindiTranslator

# Initialize all modules
print("=" * 60)
print("GOVERNMENT SERVICE ROUTING SYSTEM")
print("=" * 60)
print("\nLoading modules...")

# Create instances of each module
speech_processor = SpeechProcessor()
preprocessor = TextPreprocessor()
classifier = IntentClassifier()
router = PortalRouter()
translator = HindiTranslator()

# Train the classifier if not already trained
if not classifier.is_trained:
    print("\n" + "=" * 60)
    print("TRAINING INTENT CLASSIFIER")
    print("=" * 60)
    classifier.train("data/complaints.csv")
else:
    print("\n✅ Classifier already trained!")

# Get list of states for dropdown
STATES = router.get_all_states()
print(f"\n📍 Available states: {', '.join(STATES)}")
print("\n✅ System ready! Supports Hindi and English input")
print("=" * 60)

def find_free_port():
    """Find a free port to use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

def process_complaint(audio=None, text=None, state="Uttarakhand"):
    """
    Main function to process complaints (supports Hindi/English)
    """
    result = {
        'success': True,
        'intent': '',
        'confidence': 0,
        'department': '',
        'link': '',
        'state_used': state
    }
    
    try:
        # Step 1: Get input text (from audio or direct text)
        if audio is not None:
            # Process audio input
            print("\n Processing audio input...")
            
            if isinstance(audio, str):
                # Audio file path
                text = speech_processor.transcribe(audio)
            elif isinstance(audio, tuple) and len(audio) == 2:
                # Microphone input
                text = speech_processor.transcribe_microphone(audio)
            else:
                text = ""
        else:
            # Text input
            if not text or text.strip() == "":
                return format_error_output("Please enter your complaint")
        
        print(f"Input received: {text}")
        
        # Step 2: Detect language and translate if needed
        english_text, detected_lang = translator.process_text(text)
        
        if detected_lang == 'hi':
            print(f"Translated: {english_text}")
        
        # Step 3: Preprocess the English text
        processed_text = preprocessor.process(english_text)
        
        # Step 4: Classify intent
        intent, confidence = classifier.predict(processed_text)
        result['intent'] = intent
        result['confidence'] = round(confidence * 100, 2)
        print(f"Intent: {intent} ({result['confidence']}% confidence)")
        
        # Step 5: Get portal information
        portal_info = router.get_portal(intent, state)
        result['department'] = portal_info['department']
        result['link'] = portal_info['link']
        result['state_used'] = portal_info.get('state', state)
        
        # Step 6: Format the output
        return format_output(result)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return format_error_output("An error occurred. Please try again.")

def format_output(result):
    """
    Format the result in a clean, professional style
    """
    # Determine confidence level text
    confidence = result['confidence']
    if confidence >= 80:
        confidence_level = "High"
    elif confidence >= 60:
        confidence_level = "Medium"
    else:
        confidence_level = "Low"
    
    # Clean, professional HTML output
    output = f"""
    <div style="font-family: 'Segoe UI', Arial, sans-serif; max-width: 600px; margin: 0 auto;">
        
        <div style="background-color: #f8f9fa; border-left: 4px solid #1a73e8; padding: 20px; margin-bottom: 20px; border-radius: 0 8px 8px 0;">
            <div style="display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 15px;">
                <span style="font-size: 14px; color: #5f6368;">ANALYSIS RESULT</span>
                <span style="font-size: 12px; color: #1a73e8; background-color: #e8f0fe; padding: 4px 12px; border-radius: 16px;">{result['state_used']}</span>
            </div>
            
            <div style="margin-bottom: 25px;">
                <div style="font-size: 13px; color: #5f6368; margin-bottom: 5px;">Detected Issue</div>
                <div style="font-size: 28px; font-weight: 500; color: #202124; letter-spacing: -0.5px;">{result['intent']}</div>
            </div>
            
            <div style="margin-bottom: 25px;">
                <div style="font-size: 13px; color: #5f6368; margin-bottom: 5px;">Confidence Score</div>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="font-size: 24px; font-weight: 500; color: #202124;">{result['confidence']}%</div>
                    <div style="font-size: 13px; color: #5f6368; background-color: #f1f3f4; padding: 4px 12px; border-radius: 16px;">{confidence_level}</div>
                </div>
            </div>
        </div>
        
        <div style="background-color: white; border: 1px solid #e0e0e0; border-radius: 8px; padding: 20px;">
            <div style="font-size: 14px; color: #1a73e8; margin-bottom: 15px; letter-spacing: 0.5px;">RECOMMENDED DEPARTMENT</div>
            <div style="font-size: 20px; font-weight: 500; color: #202124; margin-bottom: 15px;">{result['department']}</div>
            
            <div style="border-top: 1px solid #e0e0e0; padding-top: 20px;">
                <a href="{result['link']}" target="_blank" style="
                    background-color: #1a73e8;
                    color: white;
                    padding: 12px 30px;
                    text-decoration: none;
                    border-radius: 6px;
                    font-size: 14px;
                    font-weight: 500;
                    display: inline-block;
                    border: none;
                    cursor: pointer;
                    transition: background-color 0.2s;
                " onmouseover="this.style.backgroundColor='#1557b0'" onmouseout="this.style.backgroundColor='#1a73e8'">
                    ACCESS PORTAL
                </a>
            </div>
        </div>
        
    </div>
    """
    
    return output

def format_error_output(message):
    """Format error messages in clean style"""
    return f"""
    <div style="font-family: 'Segoe UI', Arial, sans-serif; max-width: 600px; margin: 0 auto;">
        <div style="background-color: #fef7f7; border-left: 4px solid #d93025; padding: 20px; border-radius: 0 8px 8px 0;">
            <div style="color: #d93025; font-size: 14px; font-weight: 500; margin-bottom: 8px;">Unable to Process</div>
            <div style="color: #202124; font-size: 16px;">{message}</div>
        </div>
    </div>
    """

# Custom CSS for a professional look
custom_css = """
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 20px !important;
    background-color: #ffffff !important;
}
.gr-box {
    border: 1px solid #e0e0e0 !important;
    border-radius: 8px !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.02) !important;
}
.gr-button-primary {
    background-color: #1a73e8 !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 12px 24px !important;
    font-weight: 500 !important;
    transition: background-color 0.2s !important;
}
.gr-button-primary:hover {
    background-color: #1557b0 !important;
}
.gr-button-secondary {
    background-color: #f1f3f4 !important;
    border: 1px solid #e0e0e0 !important;
    color: #202124 !important;
    border-radius: 6px !important;
    padding: 12px 24px !important;
    font-weight: 500 !important;
}
.gr-button-secondary:hover {
    background-color: #e8eaed !important;
}
.gr-tabs {
    border: none !important;
}
.gr-tabs .tab-nav {
    background-color: #f8f9fa !important;
    border-bottom: 2px solid #e0e0e0 !important;
}
.gr-tabs .tab-nav button {
    color: #5f6368 !important;
    font-weight: 500 !important;
    border: none !important;
    background: transparent !important;
    padding: 12px 24px !important;
}
.gr-tabs .tab-nav button.selected {
    color: #1a73e8 !important;
    border-bottom: 2px solid #1a73e8 !important;
}
.gr-input, .gr-dropdown, .gr-textarea {
    border: 1px solid #e0e0e0 !important;
    border-radius: 6px !important;
    padding: 10px !important;
    font-family: 'Segoe UI', Arial, sans-serif !important;
}
.gr-input:focus, .gr-dropdown:focus, .gr-textarea:focus {
    border-color: #1a73e8 !important;
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(26,115,232,0.1) !important;
}
label {
    color: #5f6368 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    margin-bottom: 6px !important;
}
"""

# Create Gradio interface
with gr.Blocks(css=custom_css, title="Government Service Routing System") as demo:
    
    # Centered Header
    with gr.Row():
        with gr.Column():
            gr.HTML("""
            <div style="text-align: center; padding: 40px 0 30px 0;">
                <h1 style="color: #202124; font-size: 36px; font-weight: 500; letter-spacing: -0.5px; margin-bottom: 10px; font-family: 'Segoe UI', Arial, sans-serif;">
                    Government Service Routing System
                </h1>
                <p style="color: #5f6368; font-size: 16px; font-weight: 400; margin: 0;">
                    Find the correct department for your grievance
                </p>
            </div>
            """)
    
    with gr.Row(equal_height=True):
        with gr.Column(scale=5):
            # Input Section
            with gr.Tabs():
                with gr.TabItem("Voice Input"):
                    audio_input = gr.Audio(
                        type="numpy",
                        label="",
                        interactive=True
                    )
                
                with gr.TabItem("Text Input"):
                    text_input = gr.Textbox(
                        lines=3,
                        placeholder="Type your complaint here...",
                        label=""
                    )
            
            # Location Selection
            state_dropdown = gr.Dropdown(
                choices=STATES,
                value="Uttarakhand",
                label="Select your state"
            )
            
            # Action Buttons
            with gr.Row():
                submit_btn = gr.Button(
                    "Process Complaint",
                    variant="primary",
                    size="lg"
                )
                clear_btn = gr.Button(
                    "Clear",
                    variant="secondary"
                )
    
    with gr.Row():
        with gr.Column():
            # Output Section
            output_display = gr.HTML(
                value="""
                <div style="text-align: center; padding: 60px 0; color: #9aa0a6; font-family: 'Segoe UI', Arial, sans-serif;">
                    <p style="font-size: 15px;">Enter your complaint to begin</p>
                </div>
                """
            )
    
    # Examples Section
    gr.Markdown("""
    <div style="margin: 30px 0 20px 0;">
        <p style="color: #5f6368; font-size: 13px; font-weight: 500; margin-bottom: 15px;">COMMON EXAMPLES</p>
    </div>
    """)
    
    with gr.Row():
        example1 = gr.Button("Electricity Issue")
        example2 = gr.Button("Water Supply")
        example3 = gr.Button("Healthcare")
        example4 = gr.Button("Road Problem")
    
    with gr.Row():
        example5 = gr.Button("Sanitation")
        example6 = gr.Button("Hindi Input")
    
    # Footer
    gr.HTML("""
    <div style="margin-top: 60px; padding-top: 20px; border-top: 1px solid #e0e0e0; text-align: center;">
        <p style="color: #9aa0a6; font-size: 12px; margin: 0;">
            Government Service Routing System
        </p>
    </div>
    """)
    
    # Event handlers
    submit_btn.click(
        fn=process_complaint,
        inputs=[audio_input, text_input, state_dropdown],
        outputs=[output_display]
    )
    
    clear_btn.click(
        fn=lambda: (None, "", "Uttarakhand", 
                   '<div style="text-align: center; padding: 60px 0; color: #9aa0a6; font-family: \'Segoe UI\', Arial, sans-serif;"><p style="font-size: 15px;">Enter your complaint to begin</p></div>'),
        inputs=[],
        outputs=[audio_input, text_input, state_dropdown, output_display]
    )
    
    # Example buttons
    example1.click(
        fn=lambda: "no electricity in my area",
        inputs=[],
        outputs=[text_input]
    )
    
    example2.click(
        fn=lambda: "water supply not coming",
        inputs=[],
        outputs=[text_input]
    )
    
    example3.click(
        fn=lambda: "hospital is dirty",
        inputs=[],
        outputs=[text_input]
    )
    
    example4.click(
        fn=lambda: "road has potholes",
        inputs=[],
        outputs=[text_input]
    )
    
    example5.click(
        fn=lambda: "garbage not collected",
        inputs=[],
        outputs=[text_input]
    )
    
    example6.click(
        fn=lambda: "बिजली नहीं आ रही है",
        inputs=[],
        outputs=[text_input]
    )

# Launch the app with auto-port selection
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LAUNCHING APPLICATION")
    print("=" * 60)
    
    # Try different ports if 7860 is busy
    ports_to_try = [7860, 7861, 7862, 7863, 7864, 7865]
    launched = False
    
    for port in ports_to_try:
        try:
            print(f"\nTrying port {port}...")
            demo.launch(
                share=False,
                server_name="127.0.0.1",
                server_port=port,
                debug=True,
                quiet=True
            )
            launched = True
            break
        except OSError:
            print(f"Port {port} is busy, trying next...")
            continue
    
    if not launched:
        print("\n❌ Could not find a free port. Please close other applications and try again.")
        print("Alternative: Run this command to kill processes using the port:")
        print("netstat -ano | findstr :7860")
        print("taskkill /PID [PID] /F")