# 🚀 Improvement Roadmap: From 54% to <52% SMAPE

## Current Status
- **LightGBM**: 54.00% test SMAPE
- **k-NN**: 54.73% CV SMAPE  
- **Blends created**: 9 different ensemble combinations with calibration

## 📊 Quick Wins (Already Implemented)

### ✅ 1. k-NN Baseline (DONE)
Used hnswlib for fast nearest neighbor retrieval based on concatenated embeddings.
- **File**: `knn_price.py`, `validate_knn.py`
- **Result**: 54.73% CV SMAPE

### ✅ 2. Simple Blending (DONE)
Created multiple blends of LightGBM + k-NN with different weights.
- **Files**: `blend_knn_lgb.py`
- **Best blend**: 50/50 or 60/40

### ✅ 3. Calibration & Post-Processing (DONE)
Implemented clipping and quantile mapping to match train distribution.
- **Module**: `src/calibration.py`
- **Expected gain**: 0.5-1.5% SMAPE reduction

### ✅ 4. Comprehensive Ensembles (DONE)
Generated 9 calibrated ensemble submissions ready to test.
- **Script**: `create_final_ensemble.py`
- **Top picks**: 
  - `ensemble_blend_50_50_calibrated.csv`
  - `ensemble_blend_60_40_calibrated.csv`
  - `ensemble_lgb_calibrated_conservative.csv`

---

## 🎯 Next Actions (Ordered by Impact/Effort)

### Priority 1: Test Current Ensembles
**Effort**: Low | **Expected Gain**: 1-2% SMAPE

Submit these in order:
1. `submissions/ensemble_blend_50_50_calibrated.csv` ← **START HERE**
2. `submissions/ensemble_lgb_calibrated_conservative.csv`
3. `submissions/ensemble_blend_60_40_calibrated.csv`
4. `submissions/blend_50_50.csv` (no calibration)

**Goal**: Find which calibration strategy works best on the leaderboard.

---

### Priority 2: Add Aggregate Features
**Effort**: Low | **Expected Gain**: 1-2% SMAPE

**What**: Add per-category, per-brand, per-product-type median/mean prices as features.

**How**:
```bash
# Generate aggregate features
python src/aggregate_features.py

# Retrain LightGBM with new features
python train_baseline.py  # will use train_with_aggregates.csv if available
```

**Why**: Simple lookup features that capture price ranges for different product types. Very effective for pricing tasks.

---

### Priority 3: Train Neural Fusion Model
**Effort**: Medium (requires GPU, ~30-60 min) | **Expected Gain**: 1-2% SMAPE

**What**: Small MLP that takes concatenated embeddings and predicts price with custom SMAPE loss.

**How**:
```bash
python train_neural_fusion.py
```

Then blend with existing models:
```bash
python create_final_ensemble.py  # will include neural predictions
```

**Why**: Neural nets can capture non-linear interactions that trees miss. Different error patterns = better ensemble.

---

### Priority 4: Custom SMAPE Objective for LightGBM
**Effort**: Medium | **Expected Gain**: 0.5-1% SMAPE

**What**: Train LightGBM directly optimizing SMAPE instead of MSE/MAE.

**Implementation**: See `train_lgb_smape_objective.py` (create this)

**Why**: Aligns training objective with competition metric.

---

### Priority 5: Price Range Stratification
**Effort**: High | **Expected Gain**: 2-3% SMAPE

**What**: Train separate models for different price ranges (low/mid/high).

**Files**: Already have `train_stratified_ensemble.py` and `predict_stratified.py`

**How**:
```bash
python train_stratified_ensemble.py
python predict_stratified.py
```

**Why**: Different price ranges have different error characteristics. Specialized models perform better.

---

## 📈 Expected Performance Trajectory

| Action | Cumulative SMAPE | Notes |
|--------|------------------|-------|
| **Baseline LightGBM** | 54.0% | Current |
| + Calibrated blends | **52.5-53.5%** | Quick win |
| + Aggregate features | **51.5-52.5%** | Retrain needed |
| + Neural fusion | **51.0-52.0%** | 3-model ensemble |
| + SMAPE objective | **50.5-51.5%** | Fine-tuning |
| + Stratified models | **49.0-51.0%** | Significant work |

---

## 🛠️ Tools & Files Reference

### Data Processing
- `src/feature_extractor.py` - Basic feature extraction
- `src/aggregate_features.py` - Category/brand aggregates
- `src/advanced_features.py` - Domain-specific features

### Models
- `train_baseline.py` - LightGBM training
- `knn_price.py` - k-NN retrieval baseline
- `train_neural_fusion.py` - Neural network fusion
- `train_stratified_ensemble.py` - Stratified ensemble

### Post-Processing & Ensemble
- `src/calibration.py` - Calibration utilities
- `create_final_ensemble.py` - Generate all ensemble combinations
- `blend_knn_lgb.py` - Simple 2-model blending

### Validation
- `validate_knn.py` - k-NN cross-validation
- Check CV scores during training

---

## 💡 Pro Tips

1. **Always calibrate**: Calibration is cheap and almost always helps with SMAPE.

2. **Diversify your ensemble**: Use different model types (tree-based, retrieval, neural).

3. **Watch the train distribution**: Your predictions should roughly match train statistics.

4. **Price range matters**: Low-price items and high-price items have different error patterns.

5. **Embeddings are strong**: With CLIP + sentence-transformers, you already have powerful features.

6. **Don't overtune**: Test set might have different distribution. Simple blends often work best.

---

## 🎯 Immediate Next Steps

Run this now:
```bash
# 1. Submit best calibrated blend
# Already generated - just upload to competition!

# 2. Add aggregate features and retrain
python src/aggregate_features.py

# 3. Train neural fusion (if you have GPU)
python train_neural_fusion.py

# 4. Create final ensemble with all models
python create_final_ensemble.py
```

**Expected result after these steps**: 51-52% SMAPE 🎉

---

## 📁 Submission Files Summary

Generated and ready to test:
- ✅ `submissions/ensemble_blend_50_50_calibrated.csv` ← **SUBMIT FIRST**
- ✅ `submissions/ensemble_blend_60_40_calibrated.csv`
- ✅ `submissions/ensemble_lgb_calibrated_conservative.csv`
- ✅ `submissions/ensemble_knn_calibrated.csv`
- ✅ `submissions/ensemble_lgb_knn_50_50.csv`
- ✅ `submissions/ensemble_lgb_knn_60_40.csv`
- ✅ `submissions/ensemble_lgb_knn_70_30.csv`

To be generated:
- ⏳ Neural fusion predictions (after training)
- ⏳ Three-way ensemble (LGB + k-NN + Neural)
- ⏳ Stratified ensemble (if needed)

Good luck! 🚀
