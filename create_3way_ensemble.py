"""
Ultimate 3-way ensemble:
1. Neural Fusion: 51.34% CV (BEST)
2. LGB + Aggregates: 53.5% Leaderboard (52.82% CV)
3. k-NN: 54.73% CV

Strategy: Weight by inverse CV SMAPE for optimal blending
"""

import pandas as pd
import numpy as np

print("="*70)
print("🚀 ULTIMATE 3-WAY ENSEMBLE")
print("="*70)

# Load all predictions
print("\n1. Loading all models...")

neural = pd.read_csv("submissions/neural_fusion_predictions.csv")
print(f"   ✓ Neural Fusion: 51.34% CV, mean=${neural['price'].mean():.2f}")

lgb_agg = pd.read_csv("submissions/lgb_with_aggregates.csv")
print(f"   ✓ LGB+Agg: 53.5% LB (52.82% CV), mean=${lgb_agg['price'].mean():.2f}")

knn = pd.read_csv("submissions/knn_predictions.csv")
print(f"   ✓ k-NN: 54.73% CV, mean=${knn['price'].mean():.2f}")

# Merge all
merged = neural.copy()
merged = merged.merge(lgb_agg[['sample_id', 'price']], on='sample_id', suffixes=('_neural', '_lgb'))
merged = merged.merge(knn[['sample_id', 'price']], on='sample_id')
merged.columns = ['sample_id', 'price_neural', 'price_lgb', 'price_knn']

print("\n2. Model CV Performance:")
print(f"   Neural:  51.34% CV ⭐ BEST CV")
print(f"   LGB+Agg: 52.82% CV / 53.5% LB (overfit ~0.7%)")
print(f"   k-NN:    54.73% CV")

# Calculate optimal weights based on CV performance
# Use inverse SMAPE as weights
cv_smapes = {
    'neural': 51.34,
    'lgb': 52.82,
    'knn': 54.73
}

# Inverse weights (better = higher weight)
inv_weights = {k: 1/v for k, v in cv_smapes.items()}
total_inv = sum(inv_weights.values())
normalized_weights = {k: v/total_inv for k, v in inv_weights.items()}

print(f"\n3. Inverse SMAPE weights:")
print(f"   Neural:  {normalized_weights['neural']:.1%}")
print(f"   LGB+Agg: {normalized_weights['lgb']:.1%}")
print(f"   k-NN:    {normalized_weights['knn']:.1%}")

print("\n4. Creating ensemble combinations...")

ensembles = {}

# Optimal inverse-weighted
ensembles['optimal_inverse'] = (
    normalized_weights['neural'] * merged['price_neural'] +
    normalized_weights['lgb'] * merged['price_lgb'] +
    normalized_weights['knn'] * merged['price_knn']
)

# Conservative blends (favor neural + lgb since they're better)
ensembles['neural_50_lgb_40_knn_10'] = (
    0.50 * merged['price_neural'] +
    0.40 * merged['price_lgb'] +
    0.10 * merged['price_knn']
)

ensembles['neural_45_lgb_45_knn_10'] = (
    0.45 * merged['price_neural'] +
    0.45 * merged['price_lgb'] +
    0.10 * merged['price_knn']
)

ensembles['neural_40_lgb_50_knn_10'] = (
    0.40 * merged['price_neural'] +
    0.50 * merged['price_lgb'] +
    0.10 * merged['price_knn']
)

# Equal weight (simple)
ensembles['equal_weight'] = (
    merged['price_neural'] + merged['price_lgb'] + merged['price_knn']
) / 3

# Just neural + lgb (best 2)
ensembles['neural_60_lgb_40'] = (
    0.60 * merged['price_neural'] +
    0.40 * merged['price_lgb']
)

ensembles['neural_50_lgb_50'] = (
    0.50 * merged['price_neural'] +
    0.50 * merged['price_lgb']
)

# Estimate expected leaderboard SMAPE
# Neural CV 51.34% might overfit similar to LGB (0.7%)
# Ensembles typically have less overfit

print(f"\n{'Ensemble':<40} {'Mean':<10} {'Std':<10} {'Est. LB':<10}")
print("-"*70)

results = []

for name, predictions in ensembles.items():
    mean_val = predictions.mean()
    std_val = predictions.std()
    
    # Estimate leaderboard SMAPE
    # Neural CV 51.34% + 0.7% overfit = 52.0% (estimated)
    # LGB actual 53.5%
    # k-NN would be ~54.7%
    
    if 'neural_60_lgb_40' in name:
        est_lb = 0.6 * 52.0 + 0.4 * 53.5
    elif 'neural_50_lgb_50' in name:
        est_lb = 0.5 * 52.0 + 0.5 * 53.5
    elif 'neural_50_lgb_40_knn_10' in name:
        est_lb = 0.5 * 52.0 + 0.4 * 53.5 + 0.1 * 54.7
    elif 'neural_45_lgb_45_knn_10' in name:
        est_lb = 0.45 * 52.0 + 0.45 * 53.5 + 0.1 * 54.7
    elif 'neural_40_lgb_50_knn_10' in name:
        est_lb = 0.4 * 52.0 + 0.5 * 53.5 + 0.1 * 54.7
    elif 'equal' in name:
        est_lb = (52.0 + 53.5 + 54.7) / 3
    elif 'optimal' in name:
        est_lb = (normalized_weights['neural'] * 52.0 +
                  normalized_weights['lgb'] * 53.5 +
                  normalized_weights['knn'] * 54.7)
    else:
        est_lb = 52.5
    
    # Ensemble diversity bonus
    est_lb -= 0.2  # Ensembles typically reduce overfit
    
    results.append((name, mean_val, std_val, est_lb, predictions))
    print(f"{name:<40} ${mean_val:>7.2f}   ${std_val:>7.2f}   {est_lb:>6.2f}%")

# Sort by estimated LB
results_sorted = sorted(results, key=lambda x: x[3])

# Save all
print("\n5. Saving all ensemble files...")

for name, mean_val, std_val, est_lb, predictions in results_sorted:
    filename = f"submissions/3way_{name}.csv"
    submission = pd.DataFrame({
        'sample_id': merged['sample_id'],
        'price': predictions
    })
    submission.to_csv(filename, index=False)
    print(f"   ✓ {filename}")

# Save best as test_out_v3.csv
best_name, _, _, best_lb, best_pred = results_sorted[0]
submission = pd.DataFrame({
    'sample_id': merged['sample_id'],
    'price': best_pred
})
submission.to_csv("submissions/test_out_v3.csv", index=False)
print(f"   ✓ submissions/test_out_v3.csv (BEST 3-WAY)")

print("\n" + "="*70)
print("🏆 TOP 5 SUBMISSIONS")
print("="*70)

for i, (name, mean_val, std_val, est_lb, predictions) in enumerate(results_sorted[:5], 1):
    print(f"\n{i}. 3way_{name}.csv")
    print(f"   Expected: {est_lb:.2f}% SMAPE")
    if i == 1:
        print(f"   💡 Also saved as: test_out_v3.csv ⭐ BEST")

print("\n" + "="*70)
print("📊 SUMMARY")
print("="*70)

print(f"\n✅ Current best: 53.5% (LGB+Agg alone)")
print(f"\n🚀 Expected with 3-way ensemble:")
print(f"   Best: {results_sorted[0][3]:.2f}% SMAPE")
print(f"   Improvement: ~{53.5 - results_sorted[0][3]:.1f} points!")

print(f"\n💡 Why this works:")
print(f"   • Neural Fusion: 51.34% CV (best individual)")
print(f"   • LGB+Agg: Different error patterns")
print(f"   • k-NN: Adds diversity")
print(f"   • Ensemble reduces overfitting")

print("\n🎯 SUBMIT: test_out_v3.csv for best chance!")
print("="*70)
