"""
Validate ensemble using actual OOF predictions.
This gives TRUE CV SMAPE, not estimates!
"""

import pandas as pd
import numpy as np

def smape(y_true, y_pred):
    return np.mean(200 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))

print("="*70)
print("VALIDATING ENSEMBLE WITH ACTUAL OOF PREDICTIONS")
print("="*70)

# Load OOF predictions
print("\n1. Loading OOF predictions...")

lgb_oof = pd.read_csv("submissions/lgb_with_aggregates_oof.csv")
neural_oof = pd.read_csv("submissions/neural_fusion_oof.csv")

# Merge on sample_id to align
merged = lgb_oof.merge(neural_oof, on='sample_id', suffixes=('_lgb', '_neural'))

print(f"   Merged {len(merged)} OOF samples")
print(f"   Columns: {merged.columns.tolist()}")

# Extract predictions
y_true = merged['price_actual_lgb'].values  # Should be same in both
y_lgb = merged['price_pred_lgb'].values
y_neural = merged['price_pred_neural'].values

# Verify they're the same
if 'price_actual_neural' in merged.columns:
    assert np.allclose(merged['price_actual_lgb'], merged['price_actual_neural']), "Actuals don't match!"

print(f"\n2. Individual OOF SMAPE:")

lgb_smape = smape(y_true, y_lgb)
neural_smape = smape(y_true, y_neural)

print(f"   LGB+Agg:      {lgb_smape:.2f}%")
print(f"   Neural:       {neural_smape:.2f}%")

# Test different blend weights
print(f"\n3. Testing ensemble weights...")

blend_configs = [
    # 2-way blends
    ("Neural 60% + LGB 40%", 0.60, 0.40),
    ("Neural 55% + LGB 45%", 0.55, 0.45),
    ("Neural 50% + LGB 50%", 0.50, 0.50),
    ("Neural 45% + LGB 55%", 0.45, 0.55),
    ("Neural 40% + LGB 60%", 0.40, 0.60),
    ("Neural 70% + LGB 30%", 0.70, 0.30),
    ("Neural 65% + LGB 35%", 0.65, 0.35),
]

results = []

print(f"\n{'Configuration':<30} {'OOF SMAPE':<12} {'vs LGB':<10} {'vs Neural':<10}")
print("-"*70)

for name, w_neural, w_lgb in blend_configs:
    ensemble_pred = w_neural * y_neural + w_lgb * y_lgb
    ensemble_smape = smape(y_true, ensemble_pred)
    
    improvement_lgb = lgb_smape - ensemble_smape
    improvement_neural = neural_smape - ensemble_smape
    
    results.append({
        'name': name,
        'weights': (w_neural, w_lgb),
        'smape': ensemble_smape,
        'vs_lgb': improvement_lgb,
        'vs_neural': improvement_neural
    })
    
    print(f"{name:<30} {ensemble_smape:>6.2f}%    {improvement_lgb:>+6.2f}    {improvement_neural:>+6.2f}")

# Find best
results_sorted = sorted(results, key=lambda x: x['smape'])
best = results_sorted[0]

print("\n" + "="*70)
print("BEST ENSEMBLE FOUND")
print("="*70)

print(f"\n🏆 Configuration: {best['name']}")
print(f"   Weights: {best['weights'][0]:.0%} Neural + {best['weights'][1]:.0%} LGB")
print(f"   OOF SMAPE: {best['smape']:.2f}%")
print(f"   vs LGB+Agg: {best['vs_lgb']:+.2f} points")
print(f"   vs Neural: {best['vs_neural']:+.2f} points")

# Estimate leaderboard performance
# LGB: 52.82% CV → 53.5% LB (overfit 0.68%)
# Neural: 51.34% CV → ? LB
# Ensemble: should have less overfit

print(f"\n" + "="*70)
print("LEADERBOARD ESTIMATE")
print("="*70)

# Assume similar overfit pattern
lgb_overfit = 53.5 - 52.82  # 0.68%
assumed_ensemble_overfit = 0.5  # Ensembles typically overfit less

estimated_lb = best['smape'] + assumed_ensemble_overfit

print(f"\nOOF CV SMAPE: {best['smape']:.2f}%")
print(f"Assumed overfit: +{assumed_ensemble_overfit:.2f}%")
print(f"Expected LB: {estimated_lb:.2f}%")
print(f"\nImprovement from 53.5%: {53.5 - estimated_lb:+.2f} points")

# Create optimal ensemble submission
print(f"\n" + "="*70)
print("CREATING OPTIMAL ENSEMBLE")
print("="*70)

lgb_test = pd.read_csv("submissions/lgb_with_aggregates.csv")
neural_test = pd.read_csv("submissions/neural_fusion_predictions.csv")

merged_test = lgb_test.merge(neural_test, on='sample_id', suffixes=('_lgb', '_neural'))

optimal_pred = best['weights'][0] * merged_test['price_neural'] + best['weights'][1] * merged_test['price_lgb']

submission = pd.DataFrame({
    'sample_id': merged_test['sample_id'],
    'price': optimal_pred
})

submission.to_csv("submissions/OPTIMAL_ENSEMBLE.csv", index=False)

print(f"\n✅ Saved: submissions/OPTIMAL_ENSEMBLE.csv")
print(f"   Configuration: {best['name']}")
print(f"   Expected LB: {estimated_lb:.2f}%")

# Statistics
print(f"\n   Prediction stats:")
print(f"   Mean: ${optimal_pred.mean():.2f}")
print(f"   Median: ${optimal_pred.median():.2f}")
print(f"   Std: ${optimal_pred.std():.2f}")

print("\n" + "="*70)
print("TOP 5 ENSEMBLE CONFIGURATIONS")
print("="*70)

for i, r in enumerate(results_sorted[:5], 1):
    print(f"\n{i}. {r['name']}")
    print(f"   OOF SMAPE: {r['smape']:.2f}%")
    print(f"   Est. LB: {r['smape'] + assumed_ensemble_overfit:.2f}%")

print("\n" + "="*70)
print("🎯 RECOMMENDATION")
print("="*70)

print(f"\nSubmit: submissions/OPTIMAL_ENSEMBLE.csv")
print(f"Expected: {estimated_lb:.2f}% SMAPE")
print(f"Current best: 53.5%")
print(f"Improvement: {53.5 - estimated_lb:+.2f} points")

print("\n💡 This is validated on actual OOF predictions!")
print("   Much more reliable than estimates.")
print("="*70)
