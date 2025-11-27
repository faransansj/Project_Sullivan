# Models Directory

This directory stores trained model checkpoints.

**⚠️ Model files (.pth, .pt, .ckpt) are NOT tracked by Git.**

## Naming Convention

```
{phase}_{model_type}_{version}_{date}.pth

Examples:
- phase2_bilstm_baseline_20251125.pth
- phase2_conformer_v2_20251201.pth
```

## Loading Models

```python
import torch
from src.modeling.articulation_predictor import BiLSTMArticulationPredictor

model = BiLSTMArticulationPredictor()
model.load_state_dict(torch.load('models/phase2_bilstm_baseline_20251125.pth'))
model.eval()
```
