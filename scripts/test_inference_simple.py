import numpy as np
import scipy.io.wavfile as wav
from src.inference.engine import InferenceEngine
import os

# 1. Create Dummy Audio
sr = 16000
t = np.linspace(0, 2, 2 * sr) # 2 seconds
audio = 0.5 * np.sin(2 * np.pi * 440 * t)
wav_path = "test_audio.wav"
wav.write(wav_path, sr, (audio * 32767).astype(np.int16))
print(f"Created dummy audio: {wav_path}")

# 2. Init Engine
model_path = "models/transformer/final_model.ckpt"
config_path = "configs/transformer_config.yaml"
stats_path = "data/processed/stats_geometric.json"

print("Initializing Engine...")
try:
    engine = InferenceEngine(model_path, config_path, stats_path)
except Exception as e:
    print(f"Failed to init engine: {e}")
    exit(1)

# 3. Test Prediction
print("Running prediction...")
try:
    preds = engine.predict(wav_path)
    print(f"Prediction shape: {preds.shape}")
    if preds.shape[1] != 14:
        print("ERROR: Expected 14 parameters.")
    else:
        print("Prediction dimension check PASS.")
except Exception as e:
    print(f"Prediction failed: {e}")
    import traceback
    traceback.print_exc()

# 4. Test Video Generation
print("Generating video...")
out_vid = "test_viz.mp4"
try:
    res = engine.generate_video(wav_path, out_vid)
    if res and os.path.exists(res):
        print(f"Video generated successfully: {res}")
    else:
        print("Video generation failed (no output file).")
except Exception as e:
    print(f"Video generation failed: {e}")
    import traceback
    traceback.print_exc()

# Cleanup
if os.path.exists(wav_path):
    os.remove(wav_path)
