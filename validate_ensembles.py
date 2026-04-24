import pandas as pd
import numpy as np

print("="*70)
print("VALIDATION: Checking Calibrated Ensemble Quality")
print("="*70)

# Load train data
train = pd.read_csv('student_resource/dataset/train.csv')

print("\nTRAIN DATA STATISTICS:")
print(f"  Count:  {len(train)}")
print(f"  Mean:   ${train['price'].mean():.2f}")
print(f"  Median: ${train['price'].median():.2f}")
print(f"  Std:    ${train['price'].std():.2f}")
print(f"  Min:    ${train['price'].min():.2f}")
print(f"  Max:    ${train['price'].max():.2f}")
print(f"  P1:     ${train['price'].quantile(0.01):.2f}")
print(f"  P99:    ${train['price'].quantile(0.99):.2f}")

print("\n" + "="*70)
print("TOP 5 SUBMISSION FILES:")
print("="*70)

files = [
    'ensemble_blend_50_50_calibrated.csv',
    'ensemble_lgb_calibrated_conservative.csv',
    'ensemble_blend_60_40_calibrated.csv',
    'ensemble_lgb_knn_50_50.csv',
    'ensemble_knn_calibrated.csv'
]

results = []
for f in files:
    try:
        df = pd.read_csv(f'submissions/{f}')
        prices = df['price']
        results.append({
            'file': f,
            'mean': prices.mean(),
            'median': prices.median(),
            'std': prices.std(),
            'min': prices.min(),
            'max': prices.max(),
            'count': len(prices)
        })
    except Exception as e:
        print(f"\n⚠️  Could not load {f}: {e}")

print()
for i, r in enumerate(results, 1):
    print(f"{i}. {r['file']}")
    print(f"   Mean: ${r['mean']:.2f} | Median: ${r['median']:.2f} | Std: ${r['std']:.2f}")
    print(f"   Range: ${r['min']:.2f} - ${r['max']:.2f} | Count: {r['count']}")
    
    # Calculate difference from train
    diff_mean = abs(r['mean'] - train['price'].mean()) / train['price'].mean() * 100
    diff_median = abs(r['median'] - train['price'].median()) / train['price'].median() * 100
    print(f"   Δ from train: mean {diff_mean:.1f}%, median {diff_median:.1f}%")
    
    # Quality check
    if diff_mean < 10 and diff_median < 20:
        print("   ✅ GOOD - Close to train distribution")
    elif diff_mean < 20:
        print("   ⚠️  OK - Somewhat different from train")
    else:
        print("   ⚠️  WARNING - Very different from train")
    print()

print("="*70)
print("RECOMMENDATION BASED ON VALIDATION")
print("="*70)

if results:
    # Find best match to train distribution
    best = min(results, key=lambda x: abs(x['mean'] - train['price'].mean()))
    print(f"\nBest match to train distribution: {best['file']}")
    print(f"  Mean difference: {abs(best['mean'] - train['price'].mean()):.2f}")
    
    print("\n🎯 SUBMISSION ORDER:")
    print("  1. ensemble_blend_50_50_calibrated.csv (balanced)")
    print("  2. ensemble_lgb_calibrated_conservative.csv (conservative)")
    print("  3. ensemble_blend_60_40_calibrated.csv (favor LightGBM)")
    
print("\n" + "="*70)
