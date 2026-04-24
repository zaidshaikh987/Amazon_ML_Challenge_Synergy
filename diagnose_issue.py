import pandas as pd
import numpy as np

print("="*70)
print("DIAGNOSING THE 62% SMAPE ISSUE")
print("="*70)

train = pd.read_csv('student_resource/dataset/train.csv')

files_to_check = [
    'test_out.csv',                              # Your original 54% SMAPE
    'blend_50_50.csv',                           # Simple blend
    'ensemble_blend_50_50_calibrated.csv',       # The one that got 62%
    'ensemble_lgb_calibrated_conservative.csv',  # Another calibrated
]

print("\nTrain data statistics:")
print(f"  Min: ${train['price'].min():.2f}")
print(f"  Max: ${train['price'].max():.2f}")
print(f"  Mean: ${train['price'].mean():.2f}")
print(f"  Median: ${train['price'].median():.2f}")
print(f"  Std: ${train['price'].std():.2f}")

print("\n" + "="*70)
print("PREDICTION FILE COMPARISON")
print("="*70)

for filename in files_to_check:
    print(f"\n{filename}:")
    try:
        df = pd.read_csv(f'submissions/{filename}')
        prices = df['price']
        
        print(f"  Count: {len(prices)}")
        print(f"  Min: ${prices.min():.2f}")
        print(f"  Max: ${prices.max():.2f}")
        print(f"  Mean: ${prices.mean():.2f}")
        print(f"  Median: ${prices.median():.2f}")
        print(f"  Std: ${prices.std():.2f}")
        print(f"  Has negatives: {(prices < 0).any()}")
        print(f"  Has NaN: {prices.isna().any()}")
        print(f"  Count < $0.10: {(prices < 0.10).sum()}")
        print(f"  Count > $1000: {(prices > 1000).sum()}")
        
        # Check if values look suspicious
        if prices.min() < 0:
            print("  ⚠️  WARNING: Negative prices!")
        if (prices > train['price'].max() * 2).any():
            print("  ⚠️  WARNING: Very high prices!")
        if prices.std() > train['price'].std() * 3:
            print("  ⚠️  WARNING: Very high variance!")
            
    except Exception as e:
        print(f"  ERROR: {e}")

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)

print("\nThe calibration likely made predictions worse.")
print("Let's use UNCALIBRATED versions instead:")
print("\n1. test_out.csv (your original 54% SMAPE)")
print("2. blend_50_50.csv (simple LGB + k-NN blend)")
print("3. knn_predictions.csv (k-NN only)")
print("\nCalibration hurt performance - skip it for now.")
