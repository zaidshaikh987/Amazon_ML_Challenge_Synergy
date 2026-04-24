# 🚀 Quick Start Guide - Model Improvements

## ⚡ Option 1: Run Everything Automatically (RECOMMENDED)

```bash
# Run the full improvement pipeline
python run_all_improvements.py

# Or skip neural fusion (if no GPU)
python run_all_improvements.py --skip-neural
```

**What it does:**
- Adds aggregate features (category, brand, product type)
- Optionally retrains LightGBM with new features
- Optionally trains neural fusion model
- Creates 9+ calibrated ensemble submissions
- Shows recommendations

**Time:** 5-90 minutes depending on options selected

---

## 🎯 Option 2: Run Individual Steps

### Step 1: Add Aggregate Features (5 min)
```bash
python src/aggregate_features.py
```
Creates `train_with_aggregates.csv` and `test_with_aggregates.csv`

### Step 2: Create Calibrated Ensembles (1 min)
```bash
python create_final_ensemble.py
```
Generates 9 ensemble submissions with calibration

### Step 3 (Optional): Train Neural Fusion (30-60 min, GPU recommended)
```bash
python train_neural_fusion.py
```
Creates neural network predictions

### Step 4: Retrain LightGBM with Aggregate Features (15-30 min)
```bash
python train_baseline.py
# (modify to use train_with_aggregates.csv)
```

---

## 📁 Ready-to-Submit Files

**Already generated and ready to test:**

### 🥇 Top Priority (Submit These First)
```
1. submissions/ensemble_blend_50_50_calibrated.csv
   → 50% LightGBM + 50% k-NN + calibration
   → Expected: 52-53% SMAPE
   
2. submissions/ensemble_lgb_calibrated_conservative.csv
   → LightGBM with conservative calibration
   → Expected: 52-53% SMAPE
   
3. submissions/ensemble_blend_60_40_calibrated.csv
   → 60% LightGBM + 40% k-NN + calibration
   → Expected: 52-53% SMAPE
```

### 🥈 Secondary Options (Test If Above Don't Improve)
```
4. submissions/ensemble_lgb_knn_50_50.csv
   → Simple 50/50 blend (no calibration)
   
5. submissions/ensemble_lgb_knn_60_40.csv
   → 60/40 blend favoring LightGBM
   
6. submissions/ensemble_knn_calibrated.csv
   → k-NN with calibration
```

---

## 📊 Current Performance

| Model | SMAPE | Notes |
|-------|-------|-------|
| LightGBM | 54.00% | Baseline |
| k-NN | 54.73% | Similar to LightGBM |
| **Target** | **51-53%** | With calibrated blends |

---

## 🔍 Check Available Submissions

```bash
# List all ensemble submissions
Get-ChildItem submissions\ensemble_*.csv | Select-Object Name

# Or on Linux/Mac
ls -lh submissions/ensemble_*.csv
```

---

## 💡 Understanding the Improvements

### What is Calibration?
- Adjusts predictions to match training distribution
- Clips extreme values to reasonable ranges
- Reduces large relative errors → improves SMAPE
- **Expected gain:** 0.5-1.5% SMAPE

### What is Blending?
- Combines predictions from different models
- Diversifies error patterns
- k-NN captures local patterns, LightGBM captures global patterns
- **Expected gain:** 1-2% SMAPE

### What are Aggregate Features?
- Median/mean prices per category, brand, product type
- Acts as "price lookup table" for similar items
- High impact, low effort feature
- **Expected gain:** 1-2% SMAPE (when retraining)

### What is Neural Fusion?
- Small MLP trained on embeddings with SMAPE loss
- Captures non-linear interactions
- Different error patterns than trees
- **Expected gain:** 1-2% SMAPE (in ensemble)

---

## 🎯 Expected Improvement Timeline

| Action | Expected SMAPE | Time Investment |
|--------|----------------|-----------------|
| **Current** | 54.0% | - |
| + Calibrated blends | **52.5-53.5%** | 1 min ⚡ |
| + Aggregate features | **51.5-52.5%** | 5 min ⚡ |
| + Retrain with aggregates | **51.0-52.0%** | 30 min |
| + Neural fusion | **50.5-51.5%** | 60 min |
| + Stratified ensemble | **49.0-51.0%** | 2 hours |

---

## 🐛 Troubleshooting

### "File not found" errors
Make sure you have:
- ✅ Run embeddings extraction
- ✅ Trained LightGBM (train_baseline.py)
- ✅ Run k-NN predictions (knn_price.py)

### "Out of memory" errors
- Use smaller batch size in neural fusion
- Skip neural fusion with --skip-neural
- Use fewer folds in cross-validation

### Calibration makes scores worse
- Try different calibration methods (clip, quantile, conservative)
- Some competitions prefer raw predictions
- Test multiple versions on leaderboard

---

## 📈 Advanced: Further Improvements

### Custom SMAPE Objective
```bash
# Train LightGBM directly optimizing SMAPE
# (requires custom objective implementation)
python train_lgb_smape.py
```

### Stratified Ensemble
```bash
# Train separate models for different price ranges
python train_stratified_ensemble.py
python predict_stratified.py
```

### Fine-tune Embeddings
```bash
# Fine-tune CLIP/text encoders on your data
# (requires significant GPU resources)
# See advanced_finetuning.py
```

---

## 🎓 Key Learnings

1. **Calibration is free performance** - Always try it
2. **Ensemble diverse models** - Tree + retrieval + neural
3. **Domain features matter** - Category/brand aggregates help a lot
4. **Don't overtune** - Simple blends often work best
5. **Test on leaderboard** - CV scores can be misleading

---

## 🚨 Quick Command Reference

```bash
# Full pipeline (automatic)
python run_all_improvements.py --skip-neural

# Just create ensembles (fastest)
python create_final_ensemble.py

# Add aggregate features
python src/aggregate_features.py

# Train neural fusion (requires GPU)
python train_neural_fusion.py

# Check what you have
ls submissions/ensemble_*.csv
```

---

## ✅ Submission Checklist

- [ ] Upload `ensemble_blend_50_50_calibrated.csv` to competition
- [ ] Note the leaderboard score
- [ ] Try `ensemble_lgb_calibrated_conservative.csv`
- [ ] Compare results
- [ ] If better, iterate on blend weights
- [ ] If not better, try other ensembles
- [ ] Document what works in your notebook

---

## 🎉 Expected Final Result

With all improvements:
- **Target:** 51-53% SMAPE
- **Best case:** <51% SMAPE
- **Improvement:** 2-3 percentage points
- **Leaderboard jump:** Significant

Good luck! 🚀
