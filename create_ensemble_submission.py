"""
Create the optimal ensemble submission:
65% Neural Fusion + 35% LGB with Aggregates

Also creates OOF predictions file for this ensemble.
"""

import pandas as pd
import numpy as np

def smape(y_true, y_pred):
    return np.mean(200 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))

print("="*70)
print("CREATING OPTIMAL ENSEMBLE SUBMISSION")
print("="*70)

# Configuration
NEURAL_WEIGHT = 0.65
LGB_WEIGHT = 0.35

print(f"\nConfiguration: {NEURAL_WEIGHT:.0%} Neural + {LGB_WEIGHT:.0%} LGB")

# ==========================================
# 1. CREATE TEST SUBMISSION
# ==========================================
print("\n1. Creating test submission...")

# Load test predictions
neural_test = pd.read_csv("submissions/neural_fusion_predictions.csv")
lgb_test = pd.read_csv("submissions/final_lgb_agg_only.csv")

print(f"   Neural predictions: {len(neural_test)} samples")
print(f"   LGB predictions: {len(lgb_test)} samples")

# Merge on sample_id
test_merged = neural_test.merge(lgb_test, on='sample_id', suffixes=('_neural', '_lgb'))

# Create ensemble
test_merged['price'] = (NEURAL_WEIGHT * test_merged['price_neural'] + 
                        LGB_WEIGHT * test_merged['price_lgb'])

# Create submission
ensemble_submission = test_merged[['sample_id', 'price']].copy()

# Save
submission_path = "submissions/optimal_ensemble_submission.csv"
ensemble_submission.to_csv(submission_path, index=False)

print(f"   ✅ Saved: {submission_path}")
print(f"   Samples: {len(ensemble_submission)}")
print(f"   Price range: ${ensemble_submission['price'].min():.2f} - ${ensemble_submission['price'].max():.2f}")
print(f"   Mean price: ${ensemble_submission['price'].mean():.2f}")

# ==========================================
# 2. CREATE OOF PREDICTIONS
# ==========================================
print("\n2. Creating OOF predictions...")

# Load OOF predictions
neural_oof = pd.read_csv("submissions/neural_fusion_oof.csv")
lgb_oof = pd.read_csv("submissions/lgb_with_aggregates_oof.csv")

print(f"   Neural OOF: {len(neural_oof)} samples")
print(f"   LGB OOF: {len(lgb_oof)} samples")

# Merge
oof_merged = neural_oof.merge(lgb_oof, on='sample_id', suffixes=('_neural', '_lgb'))

# Create ensemble predictions
oof_merged['price_pred'] = (NEURAL_WEIGHT * oof_merged['price_pred_neural'] + 
                             LGB_WEIGHT * oof_merged['price_pred_lgb'])

# Get actual prices (should be same in both, but let's use neural)
oof_merged['price_actual'] = oof_merged['price_actual_neural']

# Calculate OOF SMAPE
oof_smape = smape(oof_merged['price_actual'].values, oof_merged['price_pred'].values)

print(f"   ✅ OOF SMAPE: {oof_smape:.2f}%")

# Create OOF file
ensemble_oof = oof_merged[['sample_id', 'price_actual', 'price_pred']].copy()

# Save
oof_path = "submissions/optimal_ensemble_oof.csv"
ensemble_oof.to_csv(oof_path, index=False)

print(f"   ✅ Saved: {oof_path}")
print(f"   Samples: {len(ensemble_oof)}")

# ==========================================
# 3. SUMMARY
# ==========================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print(f"\n📊 Model Performance:")
print(f"   Neural Fusion:  51.34% SMAPE")
print(f"   LGB+Agg:        52.82% SMAPE")
print(f"   🏆 Ensemble:    {oof_smape:.2f}% SMAPE")

improvement_from_neural = 51.34 - oof_smape
improvement_from_lgb = 52.82 - oof_smape

print(f"\n✨ Improvements:")
print(f"   vs Neural:  {improvement_from_neural:.2f} points")
print(f"   vs LGB:     {improvement_from_lgb:.2f} points")

print(f"\n📁 Files Created:")
print(f"   1. {submission_path}")
print(f"      - Test predictions for submission")
print(f"      - {len(ensemble_submission):,} samples")
print(f"")
print(f"   2. {oof_path}")
print(f"      - Out-of-fold predictions")
print(f"      - {len(ensemble_oof):,} samples")
print(f"      - SMAPE: {oof_smape:.2f}%")

print(f"\n🎯 Expected Leaderboard Performance:")
print(f"   CV Score:    {oof_smape:.2f}%")
print(f"   Overfit:     ~0.5%")
print(f"   LB Score:    ~{oof_smape + 0.5:.2f}%")
print(f"   Improvement: {53.5 - (oof_smape + 0.5):.2f} points from baseline!")

print("\n" + "="*70)
print("✅ READY FOR SUBMISSION!")
print("="*70)

print(f"\nNext steps:")
print(f"1. Submit: submissions/optimal_ensemble_submission.csv")
print(f"2. Expected LB score: ~{oof_smape + 0.5:.2f}%")
print(f"3. This should be your best submission yet!")

print("\n" + "="*70)
