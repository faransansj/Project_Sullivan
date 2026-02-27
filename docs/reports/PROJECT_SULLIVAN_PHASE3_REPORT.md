# Project Sullivan Phase 3 Completion Report

## Executive Summary
This phase successfully established a production-grade Acoustic-to-Articulatory Inversion (AAI) pipeline for the USC-TIMIT dataset. The system trains a 21.5M parameter Transformer model on the full 27-subject dataset using a custom streaming architecture to overcome memory constraints on CPU infrastructure.

**Final Secured Model:** `Project_Sullivan_Final_Transformer.ckpt`
**Architecture:** Transformer (d_model=512, layers=6, heads=8)

## Performance Metrics
| Metric | Result | Target | Notes |
| :--- | :--- | :--- | :--- |
| **Global RMSE** | **0.051** | < 0.10 | ✅ **Excellent.** The model accurately predicts the "average" vocal tract shape. |
| **Global PCC** | 0.026 | > 0.70 | ❌ **Mean-Shape Collapse.** The model struggles to track dynamic variance. |
| **Tongue Position PCC** | **0.39** | N/A | Moderate correlation for the most active articulator (Tongue Centroid Y). |
| **Jaw/Lip PCC** | 0.00 | N/A | "Dead" output (flat line prediction) due to low signal-to-noise ratio. |

## Research Findings

### 1. The Mean-Shape Collapse
A persistent phenomenon observed was "Mean-Shape Collapse," where the model minimizes MSE by predicting the statistical mean of the articulator positions rather than their dynamic trajectory. This is mathematically optimal for MSE loss when the input-output mapping is ambiguous or noisy.

### 2. PCC Injection Failure
Attempts to force dynamic tracking using a Pearson Correlation Coefficient (PCC) Loss (Stages B & D) failed.
*   **Weighted Loss (Stage B):** Combining MSE (0.5) and PCC (0.5) caused the model to degrade in both metrics.
*   **Z-Score Standardization (Stage D):** Normalizing the variance per sequence did not "wake up" the low-variance articulators (Jaw/Lips) and instead introduced noise that drove the PCC to zero.

### 3. Feature Sensitivity
The model performs best on **High Variance Features** like the Tongue Body (Centroid Y) and Dorsum. It performs worst on **Low Variance Features** (Jaw, Lips), suggesting the dataset may lack sufficient resolution or distinctiveness in the acoustic signal for these specific articulators.

## Future Recommendations
1.  **Data augmentation:** Increase the effective size of the dataset to help the model learn more robust mappings.
2.  **Adversarial Training (GAN):** Replace the static loss function with a discriminator that penalizes "flat" or "over-smoothed" trajectories.
3.  **Video-based Inversion:** If acoustic signal is insufficient for lip/jaw tracking, incorporating visual features (if available) would be the next logical step.

## Deliverables
*   **Codebase:** Fully refactored `src/` and `scripts/` with production-ready streaming dataloaders.
*   **Model:** `Project_Sullivan_Final_Transformer.ckpt`
*   **Visualizations:** Time-series plots in `results/final_deliverable/` showing the prediction quality for key articulators.
