# Logs Directory

This directory stores experiment logs and records.

## Structure

```
logs/
├── YYYY-MM/
│   └── YYYYMMDD_researcher_name.md
└── experiments/
    └── EXP-YYYYMMDD-NN.json
```

## Experiment Log Format

See `researcher_manual.md` for the complete logging template.

### Quick Template

```json
{
  "experiment_id": "EXP-20251125-01",
  "date": "2025-11-25",
  "phase": 2,
  "milestone": "M2",
  "model": "BiLSTM_Articulation_Predictor",
  "metrics": {
    "test_rmse": 0.0891,
    "test_mae": 0.0712,
    "test_pearson_correlation": [0.78, 0.82, ...]
  }
}
```
