"""
Main Application - Government Service Routing System
Coastal Retreat Color Palette - Clean UI
"""

import gradio as gr
import os
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

# Create instances
speech_processor = SpeechProcessor()
preprocessor = TextPreprocessor()
classifier = IntentClassifier()
router = PortalRouter()
translator = HindiTranslator()

# Train classifier if needed
if not classifier.is_trained:
    print("\nTraining classifier...")
    classifier.train("data/complaints.csv")

# Get states
STATES = router.get_all_states()

def process_complaint(audio=None, text=None, state="Uttarakhand"):
    """Process complaint and return formatted result"""
    try:
        # Get input text
        if audio is not None:
            if isinstance(audio, tuple) and len(audio) == 2:
                text = speech_processor.transcribe_microphone(audio)
            else:
                text = str(audio)
        elif not text or text.strip() == "":
            return format_error_output("Please enter your complaint")
        
        # Translate if Hindi
        english_text, lang = translator.process_text(text)
        
        # Preprocess
        processed = preprocessor.process(english_text)
        
        # Classify
        intent, confidence = classifier.predict(processed)
        confidence_percent = confidence * 100
        
        # Get portal
        portal = router.get_portal(intent, state)
        
        # Format output with styling
        return format_success_output(intent, confidence_percent, portal, state)
        
    except Exception as e:
        return format_error_output(str(e))

def format_success_output(intent, confidence, portal, state):
    """Format successful result with Coastal Retreat colors"""
    
    # Determine confidence color
    if confidence >= 80:
        confidence_color = "#335765"  
        confidence_text = "High"
    elif confidence >= 60:
        confidence_color = "#74A8A4"  
        confidence_text = "Medium"
    else:
        confidence_color = "#7F543D"  
        confidence_text = "Low"
    
    return f"""
    <div style="font-family: 'Segoe UI', Arial, sans-serif; background-color: white; border-radius: 16px; padding: 30px; box-shadow: 0 10px 30px rgba(51,87,101,0.15); border: 1px solid #DBE2DC;">
        
        <!-- Header with State -->
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px; padding-bottom: 20px; border-bottom: 2px solid #335765;">
            <h2 style="color: #335765; margin: 0; font-size: 24px; font-weight: 600; letter-spacing: 0.5px;">ANALYSIS RESULT</h2>
            <span style="background-color: #335765; color: white; padding: 8px 20px; border-radius: 30px; font-size: 14px; font-weight: 500; box-shadow: 0 2px 8px rgba(51,87,101,0.2);"> {state}</span>
        </div>
        
        <!-- Intent Section with B6D9E0 background -->
        <div style="background-color: #B6D9E0; padding: 25px; border-radius: 12px; margin-bottom: 25px; border-left: 5px solid #335765;">
            <div style="color: #335765; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; font-weight: 600;">Detected Issue</div>
            <div style="font-size: 36px; font-weight: 600; color: #335765; margin-bottom: 15px;">{intent}</div>
            
            <!-- Confidence Meter -->
            <div style="margin-top: 15px;">
                <div style="display: flex; align-items: center; gap: 15px;">
                    <div style="flex-grow: 1;">
                        <div style="height: 10px; background-color: #DBE2DC; border-radius: 5px; overflow: hidden;">
                            <div style="height: 100%; width: {confidence}%; background-color: {confidence_color}; border-radius: 5px;"></div>
                        </div>
                    </div>
                    <span style="font-size: 20px; font-weight: 600; color: {confidence_color};">{confidence:.1f}%</span>
                    <span style="background-color: {confidence_color}; color: white; padding: 4px 15px; border-radius: 20px; font-size: 12px; font-weight: 500;">{confidence_text}</span>
                </div>
            </div>
        </div>
        
        <!-- Department Section with DBE2DC background -->
        <div style="background-color: #DBE2DC; padding: 25px; border-radius: 12px; margin-bottom: 25px; border-left: 5px solid #74A8A4;">
            <div style="color: #335765; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; font-weight: 600;">Recommended Department</div>
            <div style="font-size: 24px; font-weight: 500; color: #335765; margin-bottom: 10px;">{portal['department']}</div>
            <div style="color: #7F543D; font-size: 14px;">{portal.get('note', 'State-specific government department')}</div>
        </div>
        
        <!-- Portal Button with 335765 color -->
        <div style="text-align: center; margin-top: 30px;">
            <a href="{portal['link']}" target="_blank" style="
                background-color: #335765;
                color: white;
                padding: 15px 45px;
                text-decoration: none;
                border-radius: 50px;
                font-size: 16px;
                font-weight: 500;
                display: inline-block;
                border: none;
                cursor: pointer;
                box-shadow: 0 4px 15px rgba(51,87,101,0.3);
                transition: all 0.3s ease;
                letter-spacing: 0.5px;
            " onmouseover="this.style.backgroundColor='#74A8A4'; this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 20px rgba(116,168,164,0.4)';" 
               onmouseout="this.style.backgroundColor='#335765'; this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 15px rgba(51,87,101,0.3)';">
                ACCESS OFFICIAL PORTAL →
            </a>
        </div>
        
        <!-- Decorative element -->
        <div style="margin-top: 20px; text-align: center;">
            <span style="color: #74A8A4; font-size: 12px;">Government Service Routing System</span>
        </div>
        
    </div>
    """

def format_error_output(message):
    """Format error message with Coastal Retreat colors"""
    return f"""
    <div style="font-family: 'Segoe UI', Arial, sans-serif; background-color: white; border-radius: 16px; padding: 30px; box-shadow: 0 10px 30px rgba(51,87,101,0.15);">
        <div style="background-color: #DBE2DC; color: #7F543D; padding: 25px; border-radius: 12px; border-left: 5px solid #7F543D;">
            <h3 style="margin: 0 0 10px 0; font-size: 18px; color: #335765;">Unable to Process</h3>
            <p style="margin: 0; font-size: 14px;">{message}</p>
        </div>
    </div>
    """

# Custom CSS with Coastal Retreat palette
custom_css = """
/* Coastal Retreat Color Palette */
:root {
    --deep-teal: #335765;
    --medium-teal: #74A8A4;
    --light-teal: #B6D9E0;
    --off-white: #DBE2DC;
    --warm-brown: #7F543D;
}

/* Main container */
.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
    padding: 20px !important;
    background: linear-gradient(135deg, #335765, #1d3340) !important;
    min-height: 100vh !important;
    font-family: 'Segoe UI', Arial, sans-serif !important;
}

/* Header styling */
.main-header {
    background: linear-gradient(135deg, #335765, #1d3340) !important;
    padding: 50px 50px 40px 50px !important;
    border-radius: 24px !important;
    margin-bottom: 40px !important;
    box-shadow: 0 15px 35px rgba(0,0,0,0.3) !important;
    border: 1px solid var(--light-teal) !important;
}

.main-header h1 {
    color: white !important;
    font-size: 48px !important;
    font-weight: 600 !important;
    margin: 0 0 15px 0 !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.2) !important;
}

.main-header p {
    color: var(--off-white) !important;
    font-size: 20px !important;
    margin: 0 !important;
}

/* Card styling */
.gr-box, .gr-form, .gr-panel {
    background-color: white !important;
    border-radius: 20px !important;
    border: none !important;
    box-shadow: 0 10px 30px rgba(51,87,101,0.2) !important;
}

/* Tab styling */
.gr-tabs {
    background-color: white !important;
    border-radius: 20px 20px 0 0 !important;
    overflow: hidden !important;
}

.tab-nav {
    background: linear-gradient(135deg, var(--off-white), white) !important;
    padding: 12px !important;
    border-bottom: 3px solid var(--deep-teal) !important;
}

.tab-nav button {
    color: var(--deep-teal) !important;
    font-weight: 600 !important;
    padding: 12px 30px !important;
    border-radius: 30px !important;
    transition: all 0.3s !important;
    font-size: 15px !important;
}

.tab-nav button.selected {
    background: linear-gradient(135deg, var(--deep-teal), var(--medium-teal)) !important;
    color: white !important;
    box-shadow: 0 4px 12px rgba(51,87,101,0.3) !important;
}

/* Input styling */
.gr-input, .gr-textarea, .gr-dropdown {
    border: 2px solid var(--light-teal) !important;
    border-radius: 12px !important;
    padding: 14px !important;
    font-size: 15px !important;
    transition: all 0.3s !important;
    background-color: var(--off-white) !important;
}

.gr-input:focus, .gr-textarea:focus, .gr-dropdown:focus {
    border-color: var(--deep-teal) !important;
    box-shadow: 0 0 0 4px rgba(51,87,101,0.1) !important;
    outline: none !important;
    background-color: white !important;
}

/* Label styling */
label {
    color: var(--deep-teal) !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    margin-bottom: 8px !important;
}

/* Button styling */
.gr-button-primary {
    background: linear-gradient(135deg, var(--deep-teal), var(--medium-teal)) !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 16px 45px !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    color: white !important;
    box-shadow: 0 4px 15px rgba(51,87,101,0.3) !important;
    transition: all 0.3s !important;
    letter-spacing: 0.5px !important;
}

.gr-button-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(116,168,164,0.4) !important;
    background: linear-gradient(135deg, var(--medium-teal), var(--deep-teal)) !important;
}

.gr-button-secondary {
    background: linear-gradient(135deg, var(--off-white), var(--light-teal)) !important;
    border: 2px solid var(--deep-teal) !important;
    border-radius: 50px !important;
    padding: 14px 45px !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    color: var(--deep-teal) !important;
    transition: all 0.3s !important;
}

.gr-button-secondary:hover {
    background: linear-gradient(135deg, var(--light-teal), var(--off-white)) !important;
    border-color: var(--medium-teal) !important;
    color: var(--deep-teal) !important;
}

/* Audio component */
.gr-audio {
    border-radius: 50px !important;
    background: linear-gradient(135deg, var(--off-white), white) !important;
    padding: 15px !important;
    border: 2px solid var(--light-teal) !important;
}

/* Dropdown styling */
.gr-dropdown select {
    background-color: var(--off-white) !important;
    border-radius: 12px !important;
}

/* Progress bar */
.gr-progress {
    background-color: var(--light-teal) !important;
}
"""

# Create Gradio interface
with gr.Blocks(css=custom_css, title="Government Service Routing System", theme=gr.themes.Base()) as demo:
    
    # Header Section
    gr.HTML("""
    <div class="main-header" style="text-align: center;">
        <h1>JanSeva Marg</h1>
        <p>Find the correct department for your grievance</p>
    </div>
    """)
    
    # Main Content Row
    with gr.Row(equal_height=True):
        # Left Column - Input Section (WITH header box - KEPT)
        with gr.Column(scale=1, min_width=500):
            gr.HTML("""
            <div style="background: white; border-radius: 20px; padding: 25px; box-shadow: 0 10px 30px rgba(51,87,101,0.15); margin-bottom: 20px;">
                <h3 style="color: #335765; margin: 0; font-size: 22px; font-weight: 600;">Input Your Complaint</h3>
                <div style="height: 4px; width: 60px; background: #74A8A4; margin-top: 10px; border-radius: 2px;"></div>
            </div>
            """)
            
            # Input Tabs
            with gr.Tabs():
                with gr.TabItem(" Voice Input"):
                    audio_input = gr.Audio(
                        type="numpy",
                        label="",
                        interactive=True
                    )
                
                with gr.TabItem(" Text Input"):
                    text_input = gr.Textbox(
                        lines=4,
                        placeholder="Type your complaint here... (e.g., 'no electricity', 'बिजली नहीं आ रही')",
                        label=""
                    )
            
            # Location Selection
            state_dropdown = gr.Dropdown(
                choices=STATES,
                value="Uttarakhand",
                label="Select Your State"
            )
            
            # Action Buttons
            with gr.Row():
                submit_btn = gr.Button(
                    "Process Complaint",
                    variant="primary",
                    size="lg"
                )
                clear_btn = gr.Button(
                    "Clear All",
                    variant="secondary"
                )
            
        # Right Column - Output Section (WITHOUT header box - REMOVED)
        with gr.Column(scale=1, min_width=500):
            # The "Analysis Result" header box has been REMOVED
            # Only the output display remains
            output_display = gr.HTML(
                value=f"""
                <div style="font-family: 'Segoe UI', Arial, sans-serif; background-color: white; border-radius: 20px; padding: 60px 30px; text-align: center; box-shadow: 0 10px 30px rgba(51,87,101,0.15); border: 1px solid #DBE2DC;">
                    <div style="color: #74A8A4; font-size: 48px; margin-bottom: 20px;">🏛️</div>
                    <div style="color: #335765; font-size: 18px; margin-bottom: 10px; font-weight: 500;">Ready to Process</div>
                    <div style="color: #7F543D; font-size: 14px;">
                        Enter your complaint and click "Process Complaint"<br>
                        to see the analysis result here.
                    </div>
                </div>
                """
            )
    
    # Footer
    gr.HTML("""
    <div style="margin-top: 50px; padding-top: 30px; border-top: 2px solid #74A8A4; text-align: center;">
        <p style="color: #B6D9E0; font-size: 13px; margin: 0;">
            Government Service Routing System • Coastal Retreat Edition
        </p>
    </div>
    """)
    
    # Event Handlers
    submit_btn.click(
        fn=process_complaint,
        inputs=[audio_input, text_input, state_dropdown],
        outputs=[output_display]
    )
    
    clear_btn.click(
        fn=lambda: (None, "", "Uttarakhand", f"""
        <div style="font-family: 'Segoe UI', Arial, sans-serif; background-color: white; border-radius: 20px; padding: 60px 30px; text-align: center; box-shadow: 0 10px 30px rgba(51,87,101,0.15); border: 1px solid #DBE2DC;">
            <div style="color: #74A8A4; font-size: 48px; margin-bottom: 20px;">🏛️</div>
            <div style="color: #335765; font-size: 18px; margin-bottom: 10px; font-weight: 500;">Ready to Process</div>
            <div style="color: #7F543D; font-size: 14px;">
                Enter your complaint and click "Process Complaint"<br>
                to see the analysis result here.
            </div>
        </div>
        """),
        inputs=[],
        outputs=[audio_input, text_input, state_dropdown, output_display]
    )

# Auto-select port and launch
if __name__ == "__main__":
    def find_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]
    
    port = find_free_port()
    print("\n" + "=" * 60)
  
    print("=" * 60)
    print(f"\n Open this URL in your browser:")
    print(f"   http://127.0.0.1:{port}")
  
    print("=" * 60)
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        quiet=True,
        share=False
    )