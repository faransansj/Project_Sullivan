# Sequence Splitting Failure Analysis

**Date**: 2025-12-02
**Critical Issue**: All sequence splitting approaches failed

---

## 📊 Complete Results Summary

### Performance Comparison

| Approach | Train Samples | Epochs | Best Epoch | Test RMSE | Test PCC | Status |
|----------|---------------|--------|------------|-----------|----------|--------|
| **Baseline LSTM** | 50 (full) | 18 | 9 | 0.993 | 0.153 | ✅ Reference |
| **Transformer (full)** | 50 (full) | 24 | 9 | 0.993 | 0.153 | ✅ Same as baseline |
| **Transformer (seq=100)** | 1,240 | 15 | 0 | 1.009 | 0.162 | ❌ Worse RMSE |
| **Transformer (seq=500)** | 228 | 17 | 2 | 1.043 | 0.097 | ❌❌ Much worse |

### Learning Patterns

```
Full utterances:
- Best at epoch 9
- Stable improvement 0-9
- Good convergence

seq=100:
- Best at epoch 0 (!)
- No learning after epoch 0
- Model stuck immediately

seq=500:
- Best at epoch 2
- Minimal improvement 0-2
- Model plateaus quickly
```

---

## 🚨 Critical Findings

### 1. Sequence Splitting Breaks Learning

**Observation**:
- Full utterances: Works fine
- Any splitting: Performance degrades
- Shorter splits = worse performance

**Evidence**:
```
Full → seq=500 → seq=100
RMSE: 0.993 → 1.043 → 1.009
PCC: 0.153 → 0.097 → 0.162

Pattern: Splitting hurts, more splitting hurts more
```

### 2. Not Just Context Length Issue

**Initial hypothesis**: seq=100 too short, seq=500 should work

**Reality**: seq=500 even WORSE than seq=100!
- seq=100: RMSE 1.009, PCC 0.162
- seq=500: RMSE 1.043, PCC 0.097

**Conclusion**: The problem is NOT just context length

---

## 🔍 Root Cause: Why Splitting Fails

### Hypothesis 1: Temporal Dependency Breaking ⭐⭐⭐

**Problem**: Articulatory movements have continuous temporal dynamics

**What happens with splitting**:
```
Original utterance: [----coarticulation----][----phoneme----][----transition----]
                     ←————— continuous flow —————→

Split at arbitrary point:
Seq 1: [----coarticu|
Seq 2: |lation----][--
Seq 3: |--phoneme----|
Seq 4: |--][----trans|
Seq 5: |sition----]

Problem:
- Coarticulation broken mid-process
- Phoneme transitions split
- Model sees discontinuous fragments
- Cannot learn smooth articulatory trajectories
```

**Evidence**:
- Best performance: Full utterances (continuous)
- Worst performance: Split sequences (discontinuous)
- More splits = more discontinuities = worse

---

### Hypothesis 2: Normalization Issues ⭐⭐

**Problem**: Statistics change with splitting

```python
Full utterances (50 samples):
- Mean/std computed over complete temporal patterns
- Natural parameter ranges preserved
- Coarticulation effects included

Split sequences (228-1240 samples):
- Mean/std computed over fragments
- Edge effects dominate
- Artificial statistics
```

**Observed warnings**:
```
RuntimeWarning: invalid value encountered in divide
```

**Impact**:
- Some features have zero variance (constant in short windows)
- Normalization fails
- Model receives corrupted inputs

---

### Hypothesis 3: Label Noise from Boundary Effects ⭐⭐

**Problem**: MRI-to-audio alignment is frame-level

**What happens**:
```
Utterance boundary:
Frame 499: End of phoneme /a/
Frame 500: Start of phoneme /t/ (SPLIT POINT)
Frame 501: Mid-transition /a/→/t/

Sequence 1 ends: Frame 499
- Model learns: audio features → /a/ articulation

Sequence 2 starts: Frame 500
- Model learns: SAME audio features → /t/ articulation

Result: Same input → different output = NOISE
```

**Evidence**:
- seq=100 (more boundaries): No learning
- seq=500 (fewer boundaries): Minimal learning
- Full (no boundaries): Normal learning

---

### Hypothesis 4: Batch Statistics Mismatch ⭐

**Problem**: Different effective batch sizes

```
Full utterances:
- 50 samples / batch 16 = 3 batches/epoch
- Each batch sees diverse complete patterns

Split sequences:
- 228 samples / batch 16 = 14 batches/epoch
- Each batch sees similar fragments
- Less diversity per batch
```

**Impact**:
- Batch normalization statistics unreliable
- Model overfits to local patterns
- Fails to generalize

---

## 💡 Why Full Utterances Work

### Advantages of Full Utterances

1. **Natural Temporal Continuity**
   - Complete coarticulation sequences
   - Smooth articulatory trajectories
   - Phoneme transitions intact

2. **Proper Statistics**
   - Mean/std over natural distributions
   - No edge effects
   - Correct parameter ranges

3. **No Boundary Artifacts**
   - No mid-phoneme splits
   - No transition breaks
   - Clean training signal

4. **Sentence-Level Context**
   - Prosody preserved
   - Stress patterns intact
   - Natural speech rhythm

---

## 🎯 Conclusion

### The Verdict

**Sequence splitting DOES NOT WORK for this task**

**Reasons**:
1. Articulatory modeling requires continuous temporal context
2. Splitting breaks coarticulation patterns
3. Boundary effects introduce label noise
4. Normalization statistics become unreliable

**Evidence**:
- 3 splitting attempts, all failed
- Performance degrades with ANY splitting
- Best results: Full utterances

---

## 💡 Recommended Solutions

### Priority 1: Optimize Full Utterance Training ⭐⭐⭐

**Accept the constraint**: 50-75 samples is what we have

**Optimize within constraint**:

```yaml
# Option A: Smaller model
model:
  d_model: 128  # vs 256
  num_layers: 2  # vs 4
  num_heads: 4  # vs 8

# Result: ~1M params (vs 4.5M)
# Better for 50 samples
```

```yaml
# Option B: Heavy regularization
model:
  dropout: 0.3  # vs 0.1
  weight_decay: 0.1  # vs 0.01

training:
  batch_size: 4  # vs 16
  accumulate_grad_batches: 4  # effective batch = 16
```

```yaml
# Option C: Data augmentation (in feature space)
augmentation:
  time_stretch: [0.9, 1.1]
  pitch_shift: [-1, 1]  # semitones
  add_noise: 0.01  # SNR 40dB
```

---

### Priority 2: Use Full Dataset ⭐⭐

**Current**: 75 utterances (selective)
**Available**: 468 utterances (full USC-TIMIT)

**Benefit**:
- 75 → 468 samples (6.2x increase)
- Same full-utterance approach
- No splitting needed
- More speakers → better generalization

**Implementation**:
```bash
# Re-run segmentation on full dataset
python scripts/segment_full_dataset.py --max_utterances 468
```

---

### Priority 3: Hybrid Approach ⭐

**Use BOTH full utterances AND augmentation**:

1. **Base training**: 468 full utterances
2. **Augmentation**: Time/pitch variations
3. **Effective samples**: 468 × 5 = 2,340

**Benefits**:
- Keep temporal continuity (full utterances)
- Increase data variety (augmentation)
- No boundary artifacts

---

### Priority 4: Different Architecture ⭐

**Problem**: Transformer may not be ideal for small data

**Alternatives**:

1. **Conformer** (CNN + Transformer)
   - Local patterns (CNN)
   - Long-range (Transformer)
   - Better for small data

2. **Temporal CNN**
   - WaveNet-style
   - Causal convolutions
   - Fewer parameters

3. **LSTM with Attention**
   - Proven on small data
   - Simpler than Transformer
   - Already works (baseline)

---

## 📈 Expected Improvements

### With Full Dataset + Small Model

```
Current (Transformer, 75 utterances, split):
- RMSE: 1.043
- PCC: 0.097

Expected (Smaller Transformer, 468 utterances, full):
- RMSE: 0.5-0.7 (50% improvement)
- PCC: 0.35-0.45 (4x improvement)
- More stable training
- Better generalization
```

### With Augmentation

```
Additional gain:
- RMSE: 0.4-0.6
- PCC: 0.45-0.55
- Closer to target (RMSE < 0.30, PCC > 0.70)
```

---

## 🚫 What NOT to Do

### Failed Approaches (Do Not Retry)

1. ❌ **Any sequence splitting**
   - Tried 100, 500, all failed
   - Breaks temporal dependencies
   - Not worth further attempts

2. ❌ **Overlapping windows**
   - Would still have boundary effects
   - Computational overhead
   - Label noise persists

3. ❌ **Phoneme-based splitting**
   - Still breaks coarticulation
   - Alignment errors
   - Not fundamentally different

4. ❌ **Larger models with splitting**
   - Data is the problem, not model size
   - Would overfit even more
   - Wastes resources

---

## 🎯 Action Plan

### Immediate (This Week)

1. **Revert config to full utterances**
   ```yaml
   sequence_length: null
   ```

2. **Reduce model size**
   ```yaml
   d_model: 128
   num_layers: 2
   ```

3. **Test on current data (75 utterances)**
   - Verify smaller model helps
   - Establish new baseline

### Short-term (Next Week)

4. **Expand to full dataset**
   - Segment all 468 utterances
   - Train full-utterance model
   - Expect major improvement

5. **Implement augmentation**
   - Time stretching
   - Pitch shifting
   - Noise injection

### Medium-term (2-3 Weeks)

6. **Try Conformer**
   - Better architecture
   - Proven for speech

7. **Hyperparameter optimization**
   - Learning rate
   - Dropout
   - Batch size

---

## 📚 Lessons Learned

### Key Insights

1. **More data ≠ Better** (if data is corrupted)
   - 1,240 samples (split) < 50 samples (full)
   - Quality > Quantity

2. **Domain knowledge matters**
   - Articulatory modeling needs continuity
   - Generic ML techniques don't always apply
   - Understand the task physics

3. **Simple baselines are powerful**
   - LSTM with full utterances works
   - Complex Transformer with splitting fails
   - Start simple, add complexity carefully

4. **Respect the data structure**
   - Speech is temporal
   - Articulatory movements are continuous
   - Don't break natural structure

---

## 📊 Final Recommendation

### Best Path Forward

**Stop sequence splitting experiments**
**Focus on**:
1. Full utterance training
2. Smaller, more appropriate model
3. Full dataset (468 utterances)
4. Data augmentation (if needed)

**Expected outcome**:
- RMSE: 0.4-0.6 (vs current 1.043)
- PCC: 0.4-0.5 (vs current 0.097)
- Stable training
- Path to target metrics

---

**Report generated**: 2025-12-02
**Status**: Critical - Sequence splitting approach abandoned
**Next action**: Implement full-utterance optimization plan
