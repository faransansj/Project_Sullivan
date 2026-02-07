import gradio as gr
import sys
import os
import time
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path.cwd()))

try:
    from src.inference.engine import InferenceEngine
except ImportError:
    # Fallback for when running from scripts/ directory
    sys.path.insert(0, str(Path.cwd().parent))
    from src.inference.engine import InferenceEngine

# --- Configuration ---
MODEL_PATH = 'models/transformer/final_model.ckpt'
CONFIG_PATH = 'configs/transformer_config.yaml'
STATS_PATH = 'data/processed/stats_geometric.json'

# --- Global Initialization ---
logger.info("Initializing Inference Engine...")
try:
    engine = InferenceEngine(
        model_path=MODEL_PATH,
        config_path=CONFIG_PATH,
        stats_path=STATS_PATH
    )
    logger.info("Engine Initialization Complete.")
except Exception as e:
    logger.error(f"Failed to initialize engine: {e}")
    # We don't exit here so Gradio can still load and show the error in logs
    engine = None

def process_audio(audio_path):
    """
    Gradio callback:
    1. Checks inputs
    2. Calls inference engine
    3. Returns video path
    """
    if engine is None:
        return None, "Error: Model not initialized. Check logs."

    if audio_path is None:
        return None, "Please record or upload audio."
    
    logger.info(f"Received audio: {audio_path}")
    
    # Create a unique output path
    timestamp = int(time.time())
    output_video = f"vis_{timestamp}.mp4"
    
    start_time = time.time()
    try:
        # Generate visualization
        video_path = engine.generate_video(audio_path, output_path=output_video)
        
        duration = time.time() - start_time
        
        if not os.path.exists(video_path):
            return None, "Error: Video generation failed (file not found)."
            
        status_msg = f"✅ Reconstruction Successful! (Processed in {duration:.2f}s)"
        return video_path, status_msg
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return None, f"❌ Error during inference: {str(e)}"

# --- Interface Definition ---
# Use Soft theme for better aesthetics
with gr.Blocks(theme=gr.themes.Soft(), title="Project Sullivan Demo") as demo:
    
    # Header
    gr.Markdown(
        """
        # 🗣️ Project Sullivan: Acoustic-to-Articulatory Inversion
        
        **Recover vocal tract shapes directly from speech audio.**
        
        This research prototype uses a **Transformer Encoder** (trained on USC-TIMIT rtMRI data) to predict 14 geometric articulatory parameters from audio.
        """
    )
    
    with gr.Row():
        # Left Column: Input
        with gr.Column(scale=1, variant="panel"):
            gr.Markdown("### 1. Input Audio")
            gr.Markdown("Record your voice or upload a WAV/MP3 file.")
            
            audio_input = gr.Audio(
                sources=["microphone", "upload"], 
                type="filepath", 
                label="Audio Source",
                waveform_options={"sample_rate": 16000}
            )
            
            gr.Markdown("### 2. Run Inference")
            process_btn = gr.Button("🚀 Reconstruct Articulation", variant="primary", size="lg")
            
            # Status Box
            status_output = gr.Textbox(label="System Status", value="Ready.", interactive=False)

        # Right Column: Output
        with gr.Column(scale=1):
            gr.Markdown("### 3. Visualization Result")
            video_output = gr.Video(label="Predicted Articulation Animation", height=400, autoplay=True)
            
            with gr.Accordion("Technical Details", open=False):
                gr.Markdown(
                    """
                    **Model Architecture:** Transformer Encoder (4 layers, 8 heads, d_model=256)
                    **Input:** Mel-spectrogram (80 bins, 16kHz audio)
                    **Output:** 14 Geometric Parameters (Tongue, Jaw, Lips, etc.)
                    **Visualization:** Green bars represent the normalized activation (0-1) of each parameter over time.
                    """
                )

    # Footer
    gr.Markdown("---")
    gr.Markdown(
        """
        <center>
        Project Sullivan Phase 5 Demo | Powered by Gradio & PyTorch
        </center>
        """
    )

    # Event binding
    process_btn.click(
        fn=process_audio,
        inputs=[audio_input],
        outputs=[video_output, status_output]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        share=False
    )