# Phase 5: Web-based Interactive Demo Plan

**Goal**: Develop a user-friendly web application that allows users to record their voice and visualize the reconstructed vocal tract movements in real-time (or near real-time).

**Branch**: `feature/web-demo`
**Tech Stack**: Python, Gradio, PyTorch, Plotly/Matplotlib

---

## 📅 Milestones

### M1: Inference Engine Wrapper (Backend)
**Objective**: Abstract the complex preprocessing and inference logic into a clean Python API.
- [ ] Create `src/inference/engine.py`.
- [ ] Implement `InferenceEngine` class:
    - Load 21.5M Master Transformer (`last.ckpt`).
    - Load PCA model (`pca_model.npz`).
    - Implement `process_audio(audio_path) -> articulatory_video`.
    - Handle audio resampling (16kHz) and Mel-spectrogram extraction internally.

### M2: Gradio UI Prototype (Frontend)
**Objective**: Build a browser-based interface for interaction.
- [ ] Create `scripts/app.py`.
- [ ] Setup Gradio Blocks:
    - **Input**: Microphone / File Upload.
    - **Output**: Video/GIF player + Static Plots (Key Frames).
    - **Controls**: "Generate Reconstruction" button.

### M3: Visualization & Animation
**Objective**: Convert 24-dim output vectors into a smooth visual representation.
- [ ] Implement high-speed mask reconstruction (Matrix multiplication).
- [ ] Optimize GIF generation (avoid Matplotlib loop bottlenecks if possible, consider raw OpenCV/MoviePy).
- [ ] (Optional) Add "Tongue Height" and "Jaw Opening" quantitative gauges.

### M4: Optimization & Packaging
**Objective**: Ensure the demo runs smoothly on standard CPUs.
- [ ] Quantize model to INT8 (Dynamic Quantization) for speed.
- [ ] Create `Dockerfile` for easy deployment.
- [ ] Update `requirements.txt` with `gradio`, `moviepy`.

---

## 🛠️ Implementation Plan

### Step 1: Directory Setup
```bash
mkdir -p src/inference
touch src/inference/__init__.py
```

### Step 2: Dependencies
Add `gradio` and `moviepy` to `requirements.txt`.

### Step 3: Execution
Run `python scripts/app.py` to launch the local server.
