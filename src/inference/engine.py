import sys
import json
from pathlib import Path
import yaml
import torch
import numpy as np
import librosa
import cv2

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.modeling.transformer import TransformerModel

class InferenceEngine:
    def __init__(self, model_path, config_path, stats_path=None):
        print("Initializing Inference Engine...")
        self.device = torch.device('cpu') # Force CPU for inference
        
        # Load config to get model and data params
        print(f"Loading config from {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Load Normalization Stats
        self.stats = None
        if stats_path:
            print(f"Loading statistics from {stats_path}")
            with open(stats_path, 'r') as f:
                self.stats = json.load(f)
                # Convert to numpy for fast ops
                self.stats['min'] = np.array(self.stats['min'])
                self.stats['max'] = np.array(self.stats['max'])
                self.stats['mean'] = np.array(self.stats['mean'])
                self.stats['std'] = np.array(self.stats['std'])

        # Load trained Transformer model
        print(f"Loading Master Transformer from {model_path}")
        # We need to instantiate the model with the same args as training
        # But load_from_checkpoint usually handles this if hparams were saved
        self.model = TransformerModel.load_from_checkpoint(model_path, map_location=self.device)
        self.model.eval()
        print("Inference Engine Ready.")

    def _preprocess_audio(self, audio_path):
        """Load, resample, and extract Mel spectrogram."""
        # Defaults from typical config if missing
        data_conf = self.config.get('data', {})
        sr = data_conf.get('sr', 16000)
        n_fft = data_conf.get('n_fft', 512)
        hop_length = data_conf.get('hop_length', 160)
        
        # Model config usually has input_dim (n_mels)
        model_conf = self.config.get('model', {})
        n_mels = model_conf.get('input_dim', 80)

        # Load and Resample
        try:
            audio, original_sr = librosa.load(audio_path, sr=None)
        except Exception as e:
            raise ValueError(f"Failed to load audio file {audio_path}: {e}")
            
        if original_sr != sr:
            audio = librosa.resample(audio, orig_sr=original_sr, target_sr=sr)

        # Extract Mels
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Transpose to (Time, Mels)
        return mel_spec_db.T

    def _denormalize(self, predictions):
        """Denormalize predictions using loaded statistics."""
        if self.stats is None:
            # Fallback: assume output is already in reasonable range or return as is
            return predictions
        
        # Assume MinMax normalization as default for this project
        # val_denorm = val_norm * (max - min) + min
        
        # Ensure stats match dimension
        if len(self.stats['min']) != predictions.shape[1]:
            print(f"Warning: Stats dimension ({len(self.stats['min'])}) matches not pred dimension ({predictions.shape[1]})")
            # Try to match common subset if possible, or just return
            return predictions

        p_min = self.stats['min']
        p_max = self.stats['max']
        p_range = p_max - p_min
        
        # Handle zero range
        p_range[p_range == 0] = 1.0
        
        denormalized = predictions * p_range + p_min
        return denormalized

    def predict(self, audio_path):
        """
        Run full inference pipeline.
        Returns:
            np.ndarray: Denormalized parameters (Time, 14)
        """
        # Preprocess audio
        mel_features = self._preprocess_audio(audio_path)
        
        # Convert to tensor and add batch dimension
        audio_tensor = torch.FloatTensor(mel_features).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            preds_norm = self.model(audio_tensor)
        
        # Remove batch dimension
        preds_norm_np = preds_norm.squeeze(0).cpu().numpy()
        
        # Denormalize
        preds_denorm = self._denormalize(preds_norm_np)
        
        return preds_denorm

    def generate_video(self, audio_path, output_path="output.mp4"):
        """
        Creates a visualization video of the predicted parameters.
        Generates a bar chart animation.
        """
        # 1. Get Predictions
        params = self.predict(audio_path)
        
        if params is None or len(params) == 0:
            return None

        # 2. Setup Video Writer
        # Assume ~80 FPS for MRI data (typical for rtMRI)
        # Audio extraction hop_length=160, sr=16000 => 100 FPS
        # Check config for true FPS, or approx 100
        fps = 100 
        
        height, width = 480, 640
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Colors for bars
        bar_color = (0, 255, 0) # Green
        text_color = (255, 255, 255)
        bg_color = (0, 0, 0)
        
        # Determine global min/max for plotting scaling
        # (Use stats if avail, else data min/max)
        if self.stats:
            g_min = self.stats['min']
            g_max = self.stats['max']
        else:
            g_min = np.min(params, axis=0)
            g_max = np.max(params, axis=0)
            
        n_params = params.shape[1]
        bar_width = width // n_params
        
        # 3. Render Frames
        for t in range(len(params)):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Draw frame info
            cv2.putText(frame, f"Frame: {t}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            
            current_vals = params[t]
            
            for i, val in enumerate(current_vals):
                # Normalize val for display height
                # mapped [min, max] -> [0, height-50]
                val_min = g_min[i]
                val_max = g_max[i]
                val_range = val_max - val_min if val_max != val_min else 1
                
                norm_h = (val - val_min) / val_range
                bar_h = int(norm_h * (height - 60))
                bar_h = max(0, min(height-60, bar_h)) # Clip
                
                # Draw Bar
                # Top-Left: (x, height-bar_h-30)
                # Bottom-Right: (x+w, height-30)
                x = i * bar_width
                y_top = height - bar_h - 30
                y_bottom = height - 30
                
                cv2.rectangle(frame, (x, y_top), (x + bar_width - 2, y_bottom), bar_color, -1)
                
                # Draw param index
                cv2.putText(frame, str(i), (x + 5, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)

            out.write(frame)
            
        out.release()
        return output_path