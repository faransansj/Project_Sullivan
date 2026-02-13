# 🌐 Project Sullivan: Web Demo Guide (Phase 5)

This guide provides instructions on setting up and using the **Interactive Web Demo** for Acoustic-to-Articulatory Inversion (AAI).

The demo allows users to record their voice or upload audio files and visualize the predicted movement of 14 articulatory parameters (tongue, jaw, lips, etc.) in real-time.

---

## ✅ Prerequisites

Before running the demo, ensure your environment meets the following requirements:

- **OS**: Linux (Recommended), macOS, or Windows
- **Python**: 3.9 or higher
- **RAM**: 4GB+ (8GB recommended)
- **Disk**: 500MB+ for model weights and dependencies

---

## 🛠️ Installation

### 1. Set up Virtual Environment
It is highly recommended to use a virtual environment to avoid conflicts.

```bash
# Create virtual environment
python -m venv .venv

# Activate (Linux/Mac)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate
```

### 2. Install Dependencies
Install all required packages, including `gradio` and `pytorch`.

```bash
pip install -r requirements.txt
```

### 3. Verify Model Checkpoint
Ensure the trained Transformer model checkpoint is present.
- Path: `models/transformer/final_model.ckpt`
- Stats: `data/processed/stats_geometric.json`

*(Note: If these files are missing, refer to `docs/guides/DATA_DOWNLOAD_GUIDE.md` or train the model using Phase 4 scripts.)*

---

## 🚀 Running the Demo

To launch the web interface, execute the application script from the project root:

```bash
python scripts/app.py
```

**Expected Output:**
```
INFO:src.inference.engine:Initializing Inference Engine...
...
INFO:src.inference.engine:Inference Engine Ready.
Running on local URL:  http://0.0.0.0:7860
```

Open your browser and navigate to **http://localhost:7860**.

---

## 🎮 Using the Interface

The interface is divided into two main sections:

### 1. Input Audio (Left Panel)
- **Record**: Click the microphone icon to record your voice directly.
- **Upload**: Click the upload area to select a `.wav` or `.mp3` file from your computer.
- **Submit**: Click the **"🚀 Reconstruct Articulation"** button to start processing.

### 2. Visualization (Right Panel)
- After processing (approx. 1-3 seconds for short clips), a video will appear.
- **Green Bars**: Represent the activation level of 14 geometric articulatory parameters.
- **Labels**:
    - **Tongue**: Position (X, Y), Height, Fronting
    - **Jaw**: Opening degree
    - **Lips**: Aperture, Protrusion
- The video plays automatically. You can pause, scrub, or download it.

---

## 🔧 Troubleshooting

| Issue | Solution |
| :--- | :--- |
| **ModuleNotFoundError** | Run `pip install -r requirements.txt` again. Check `PYTHONPATH`. |
| **"Model not initialized"** | Verify `models/transformer/final_model.ckpt` exists. Check logs for loading errors. |
| **Slow Processing** | Processing happens on CPU by default. Short audio clips (< 10s) are recommended. |
| **No Video Output** | Ensure `ffmpeg` or `opencv-python` is installed correctly. |

---

**Project Lead**: Midori (AI Agent)
**Last Update**: Jan 2026
