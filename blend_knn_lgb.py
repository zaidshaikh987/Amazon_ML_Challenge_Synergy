import pandas as pd
import numpy as np

print("="*70)
print("BLENDING k-NN + LightGBM")
print("="*70)

# Load predictions
print("\n1. Loading predictions...")

# k-NN predictions (just generated)
knn = pd.read_csv("submissions/knn_predictions.csv")

# Try to find your best LightGBM predictions
# Check multiple possible files
lgb_files = [
    "submissions/test_out.csv",
    "submissions/test_out_01.csv", 
    "submissions/test_out_02.csv"
]

lgb = None
for file in lgb_files:
    try:
        lgb = pd.read_csv(file)
        print(f"   Loaded LightGBM from: {file}")
        break
    except:
        pass

if lgb is None:
    print("   ERROR: Could not find LightGBM predictions")
    print("   Please specify the correct file")
    exit(1)

# Merge on sample_id
print("\n2. Merging predictions...")
merged = knn.merge(lgb, on='sample_id', suffixes=('_knn', '_lgb'))

print(f"   Merged {len(merged)} samples")
print(f"   k-NN:  mean=${merged['price_knn'].mean():.2f}, std=${merged['price_knn'].std():.2f}")
print(f"   LightGBM: mean=${merged['price_lgb'].mean():.2f}, std=${merged['price_lgb'].std():.2f}")

# Try different blend weights
print("\n3. Creating blends with different weights...")

blends = [
    (0.5, 0.5, "50_50"),
    (0.4, 0.6, "40_60_favor_lgb"),
    (0.6, 0.4, "60_40_favor_knn"),
    (0.3, 0.7, "30_70_favor_lgb"),
]

for w_knn, w_lgb, name in blends:
    blended = w_knn * merged['price_knn'] + w_lgb * merged['price_lgb']
    
    result = pd.DataFrame({
        'sample_id': merged['sample_id'],
        'price': blended
    })
    
    filename = f"submissions/blend_{name}.csv"
    result.to_csv(filename, index=False)
    
    print(f"   {name}: mean=${blended.mean():.2f}, std=${blended.std():.2f}")
    print(f"      Saved to {filename}")

# Summary
print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)
print("\nSince k-NN and LightGBM have similar CV SMAPE (~54%),")
print("blending them might give 1-2 point improvement.")
print("\nTry submitting:")
print("1. submissions/blend_50_50.csv (equal weight)")
print("2. submissions/blend_40_60_favor_lgb.csv (favor LightGBM)")
print("\nExpected test SMAPE: 52-54% (hopefully better than 54%)")
print("="*70)
