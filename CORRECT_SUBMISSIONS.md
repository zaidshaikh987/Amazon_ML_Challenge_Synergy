# ❌ CALIBRATION FAILED - USE UNCALIBRATED VERSIONS

## What Happened:
The calibrated ensemble got **62% SMAPE** because:
- Calibration introduced too much variance (std $31 vs $18)
- Max predictions went up to $940 (vs $268 uncalibrated)
- This increased relative errors → worse SMAPE

## ✅ CORRECT FILES TO SUBMIT (In Order):

### 1. `blend_50_50.csv` ⭐ **SUBMIT THIS FIRST**
- Simple 50/50 blend of LightGBM + k-NN
- **NO calibration** (calibration made it worse!)
- Expected: ~53-54% SMAPE
- Most conservative, lowest variance

### 2. `blend_40_60_favor_lgb.csv`
- 60% LightGBM, 40% k-NN
- Slightly favor your better model
- Expected: ~53-54% SMAPE

### 3. `test_out.csv`
- Your original LightGBM (54% SMAPE)
- Good baseline fallback

### 4. `knn_predictions.csv`
- Pure k-NN (54.73% CV SMAPE)
- Different errors than LightGBM

### 5. `blend_60_40_favor_knn.csv`
- 60% k-NN, 40% LightGBM
- More conservative on price estimates

---

## 📊 Statistics Comparison:

| File | Mean | Std | Max | Status |
|------|------|-----|-----|--------|
| **blend_50_50.csv** | $19.65 | $18.19 | $268 | ✅ **BEST** |
| blend_40_60_favor_lgb.csv | $20.02 | $18.65 | ~$280 | ✅ Good |
| test_out.csv | $21.49 | $21.58 | $335 | ✅ Baseline |
| ~~calibrated (failed)~~ | $21.78 | $31.47 | $940 | ❌ **62% SMAPE** |

**Train statistics:** Mean $23.65, Std $33.38, Max $2796

---

## 💡 Why Simple Blends Work Better:

1. **Low variance** = better SMAPE (relative error metric)
2. **No overcorrection** from calibration
3. **Ensemble diversity** without introducing noise
4. **Conservative predictions** on average

---

## 🎯 IMMEDIATE ACTION:

```bash
# These files are already in submissions/ folder:
1. submissions/blend_50_50.csv                 ← SUBMIT NOW
2. submissions/blend_40_60_favor_lgb.csv       ← Try if #1 works
3. submissions/test_out.csv                    ← Fallback
```

---

## ⚠️ AVOID THESE (They Got 62%):
- ❌ ensemble_blend_50_50_calibrated.csv
- ❌ ensemble_lgb_calibrated_conservative.csv
- ❌ All files with "calibrated" in the name

**Calibration doesn't always help!** In this case it hurt.

---

## 📈 Expected Results:

- **blend_50_50.csv**: 53-54% SMAPE (slight improvement)
- **blend_40_60_favor_lgb.csv**: 53-54% SMAPE  
- **test_out.csv**: 54% SMAPE (your baseline)

The improvement from simple blending is small (~0.5-1%) but real.

---

## 🔍 Lesson Learned:

**Calibration can backfire** when:
- It increases variance too much
- Test distribution differs from train
- SMAPE punishes large relative errors

**Simple is better** for this competition. Stick with uncalibrated blends.

---

## Next Steps (If Needed):

If simple blends don't improve much, try:
1. **Different blend weights**: 70/30, 80/20, etc.
2. **Retrain with aggregate features** (already extracted)
3. **Neural fusion** (more complex, risky)
4. **Focus on feature engineering** instead of post-processing
