# Colab Training Issue Analysis - Sequence Splitting Problem

**Date**: 2025-12-02
**Issue**: Performance degraded after enabling sequence splitting

---

## 📊 Comparison

### Training Setup

| Metric | Run 1 (No Split) | Run 2 (Split=100) | Change |
|--------|------------------|-------------------|--------|
| Train samples | 50 | 1,240 | +2,380% |
| Val samples | 10 | 213 | +2,030% |
| Test samples | 15 | 372 | +2,380% |
| Epochs trained | 24 | 15 | Early stop |
| Best epoch | 9 | 0 | First epoch |

### Performance Results

| Metric | Run 1 (No Split) | Run 2 (Split=100) | Change |
|--------|------------------|-------------------|--------|
| **Test RMSE** | 0.993 | **1.009** | ❌ +1.6% worse |
| **Test PCC** | 0.153 | 0.162 | ✅ +5.9% better |
| **Test MAE** | 0.736 | 0.757 | ❌ +2.8% worse |
| **Val Loss (best)** | 0.868 | 1.033 | ❌ +19% worse |

---

## 🔍 Root Cause Analysis

### Issue 1: Sequence Length Too Short

**Problem**:
```
Average utterance length: ~2,500 frames (25 seconds)
Sequence length: 100 frames (1 second)
Context captured: 4% of utterance
```

**Impact**:
- Transformer sees only 1-second chunks
- Missing long-range dependencies
- Articulatory movements span multiple seconds
- Context from previous phonemes lost

**Evidence**:
- Performance got worse, not better
- No improvement after epoch 0
- Model cannot learn temporal patterns

---

### Issue 2: Training Dynamics Changed

**Batch Updates**:
```
Run 1: 50 samples / 16 batch = 3-4 updates/epoch
Run 2: 1,240 samples / 16 batch = 78 updates/epoch

Total updates in 15 epochs:
Run 1 equivalent: 45-60 updates
Run 2: 1,170 updates (20x more!)
```

**Problem**:
- Same learning rate (0.0005)
- 20x more gradient updates
- Model might be "overfitting" to short sequences
- Or underfitting due to lack of context

---

### Issue 3: Data Distribution Mismatch

**Statistics**:
```
Run 1: Normalize over 50 full utterances
- Mean/std computed from complete temporal patterns
- Natural coarticulation effects preserved

Run 2: Normalize over 1,240 short sequences
- Mean/std from fragmented chunks
- Temporal context broken
- Edge effects at sequence boundaries
```

**Warning observed**:
```python
RuntimeWarning: invalid value encountered in divide
```
- Some features have zero variance
- Normalization fails for constant sequences

---

### Issue 4: Sequence Boundary Effects

**Problem**:
```
Utterance: [phoneme1][phoneme2][phoneme3][phoneme4]...
           |------100------|------100------|------100------|

Sequence 1: [phoneme1][pho...] (incomplete context)
Sequence 2: [...neme2][pho...] (starts mid-phoneme)
Sequence 3: [...neme3][pho...] (starts mid-phoneme)
```

**Impact**:
- Sequences start/end at arbitrary points
- No phonetic boundaries respected
- Coarticulation effects split across sequences
- Model sees "meaningless" fragments

---

## 💡 Recommended Solutions

### Priority 1: Increase Sequence Length ⭐⭐⭐

**Option A: Longer sequences**
```yaml
sequence_length: 500  # 5 seconds, captures full words/phrases
```

**Expected improvement**:
- Better context for Transformer
- Captures coarticulation
- More natural temporal dependencies

**Trade-offs**:
- Fewer samples: 1,240 → ~250
- Still 5x more than original 50
- More memory per sequence

---

**Option B: Very long sequences**
```yaml
sequence_length: 1000  # 10 seconds
```

**Expected improvement**:
- Even better context
- Captures sentence-level patterns

**Trade-offs**:
- ~125 samples
- Still 2.5x more than original
- May need smaller batch size

---

**Option C: Use full utterances with padding**
```yaml
sequence_length: null  # Back to full utterances
batch_size: 4  # Smaller batch to fit in memory
accumulate_grad_batches: 4  # Effective batch = 16
```

**Reasoning**:
- Original approach might have been correct
- Articulatory movements need full context
- Use gradient accumulation for effective larger batch

---

### Priority 2: Adjust Learning Rate ⭐⭐

**With more samples, need different LR strategy**:

```yaml
# Option A: Lower learning rate
learning_rate: 0.0001  # 5x lower for 20x more updates

# Option B: Linear warmup
lr_scheduler: LinearWarmupCosineAnnealing
lr_scheduler_params:
  warmup_epochs: 5
  max_epochs: 50
```

---

### Priority 3: Smart Sequence Splitting ⭐

**Instead of fixed-length splitting, use phonetic boundaries**:

```python
# Pseudo-code
def split_at_phonetic_boundaries(utterance, target_length=500):
    """Split at silence or phoneme boundaries"""
    sequences = []
    current_seq = []

    for frame in utterance:
        current_seq.append(frame)

        if len(current_seq) >= target_length:
            # Look for next silence or low-energy frame
            if is_silence(frame) or low_energy(frame):
                sequences.append(current_seq)
                current_seq = []

    return sequences
```

---

### Priority 4: Overlapping Windows ⭐

**Add overlap to preserve context**:

```yaml
sequence_length: 200
sequence_overlap: 50  # 25% overlap

# Sequence 1: frames [0:200]
# Sequence 2: frames [150:350] (50 frame overlap)
# Sequence 3: frames [300:500]
```

**Benefits**:
- More samples
- Context preserved at boundaries
- Smoother transitions

---

## 🎯 Recommended Next Steps

### Immediate Action: Test Different Sequence Lengths

**Quick experiment on CPU**:

1. **Test sequence_length=500**
   ```bash
   # Update config
   sed -i 's/sequence_length: 100/sequence_length: 500/' configs/transformer_cpu_test.yaml
   # Run test
   python scripts/train_transformer.py --config configs/transformer_cpu_test.yaml
   ```

2. **Test sequence_length=1000**
   ```bash
   sed -i 's/sequence_length: 500/sequence_length: 1000/' configs/transformer_cpu_test.yaml
   python scripts/train_transformer.py --config configs/transformer_cpu_test.yaml
   ```

3. **Compare results**:
   - Which length gives best val_loss?
   - Does training improve beyond epoch 0?
   - Are there enough samples?

---

### Medium-term: Full Utterance Training

**Revert to full utterances with optimizations**:

```yaml
data:
  sequence_length: null  # Use full utterances

training:
  batch_size: 4  # Smaller for memory
  accumulate_grad_batches: 4  # Effective batch = 16

optimization:
  learning_rate: 0.0001  # Lower LR for stable training
```

**Expected**:
- Better context preservation
- More stable training
- Performance closer to baseline or better

---

### Long-term: Hybrid Approach

**Use both full and split sequences**:

1. **Pre-training**: Long sequences (500-1000 frames)
   - Learn basic audio-articulation mapping
   - Capture coarticulation effects

2. **Fine-tuning**: Full utterances
   - Refine with complete context
   - Learn sentence-level patterns

---

## 📈 Success Criteria

**Next run should show**:

1. **Training improvement** ✅
   - Val loss decreases over epochs
   - Not stuck at epoch 0

2. **Performance improvement** ✅
   - Test RMSE < 0.99 (better than both previous runs)
   - Test PCC > 0.20

3. **Stable convergence** ✅
   - Best epoch after 5-10 epochs
   - Smooth val_loss curve

---

## 🔬 Hypothesis

**Current hypothesis**:
- **100 frames is too short** for articulatory modeling
- Articulatory movements have temporal dependencies spanning 0.5-2 seconds
- Transformer needs at least 3-5 seconds (300-500 frames) of context
- Original "full utterance" approach was actually better for this task

**Supporting evidence**:
- Run 1 (full utterances): RMSE 0.993, best at epoch 9
- Run 2 (100-frame chunks): RMSE 1.009, best at epoch 0 (no learning!)
- CPU test (100 frames, 2 epochs): RMSE 0.980 but no improvement trend

**Test**:
- Try sequence_length = 500
- If performance improves and training is stable → hypothesis confirmed
- If still no improvement → other issues (LR, normalization, etc.)

---

## 📚 Literature Support

**From speech processing research**:
- Coarticulation effects span 100-300ms (10-30 frames at 100fps)
- Phoneme duration: 50-200ms (5-20 frames)
- Word duration: 200-1000ms (20-100 frames)
- **Sentence-level prosody: 1-5 seconds (100-500 frames)**

**Articulatory modeling papers**:
- Most successful models use 500-2000ms windows
- Full-sentence models often outperform frame-by-frame
- Transformer models need sufficient context length

**Conclusion**:
- 100 frames (1 second) is minimum for word-level context
- 500 frames (5 seconds) is better for sentence-level patterns
- Full utterances ideal for capturing complete articulatory sequences

---

**Report generated**: 2025-12-02
**Status**: Performance regression identified, solutions proposed
**Next action**: Test sequence_length = 500 on CPU first
