# ⚡ Project Sullivan: Demo Quick Start

Run the interactive Acoustic-to-Articulatory Inversion demo in **3 steps**.

### 1. Setup
```bash
# Clone & Enter
git clone https://github.com/faransansj/Project_Sullivan.git
cd Project_Sullivan

# Environment
python -m venv .venv
source .venv/bin/activate

# Dependencies (includes Gradio)
pip install -r requirements.txt
```

### 2. Launch
```bash
python scripts/app.py
```

### 3. Use
1. Open browser: **http://localhost:7860**
2. Record voice or upload audio.
3. Click **"Reconstruct"** to see the vocal tract animation.

---
*For details, see [Web Demo Guide](docs/guides/WEB_DEMO_GUIDE.md).*
