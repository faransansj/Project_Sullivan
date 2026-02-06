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
    
    try:
        # Generate visualization
        # Note: generate_video returns the path to the video
        video_path = engine.generate_video(audio_path, output_path=output_video)
        
        if not os.path.exists(video_path):
            return None, "Error: Video generation failed (file not found)."
            
        return video_path, "Reconstruction Successful!"
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return None, f"Error during inference: {str(e)}"

# --- Interface Definition ---
with gr.Blocks(title="Project Sullivan Demo") as demo:
    gr.Markdown(
        """
        # 🗣️ Project Sullivan: Acoustic-to-Articulatory Inversion
        
        **Recover vocal tract shapes from speech audio.**
        
        This demo uses a Transformer model to predict 14 geometric articulatory parameters 
        (tongue position, jaw opening, etc.) from Mel-spectrogram features.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Input Audio")
            audio_input = gr.Audio(
                sources=["microphone", "upload"], 
                type="filepath", 
                label="Microphone / Upload"
            )
            process_btn = gr.Button("🚀 Reconstruct Articulation", variant="primary")
            status_output = gr.Textbox(label="Status", interactive=False)
            
        with gr.Column(scale=1):
            gr.Markdown("### 2. Visualization")
            video_output = gr.Video(label="Predicted Articulation", height=400)
            
    gr.Markdown("---")
    gr.Markdown("*Note: This is a research prototype. The visualization shows normalized parameter values.*")

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