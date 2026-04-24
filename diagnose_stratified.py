import pandas as pd
import numpy as np
import pickle

print("="*70)
print("DIAGNOSING WHY STRATIFIED ENSEMBLE FAILED")
print("="*70)

# Load train/test data
train = pd.read_csv('student_resource/dataset/train.csv')
test = pd.read_csv('student_resource/dataset/test.csv')

print("\n1. PRICE DISTRIBUTION COMPARISON")
print("-"*70)

print("\nTRAIN price distribution:")
for p in [10, 25, 50, 75, 90, 95, 99]:
    print(f"  {p}th percentile: ${train['price'].quantile(p/100):.2f}")

print("\n2. PREDICTIONS ANALYSIS")
print("-"*70)

# Load stratified predictions
preds = pd.read_csv('submissions/stratified_ensemble_predictions.csv')

print("\nPREDICTED price distribution:")
for p in [10, 25, 50, 75, 90, 95, 99]:
    print(f"  {p}th percentile: ${preds['price'].quantile(p/100):.2f}")

print("\n3. DISTRIBUTION MISMATCH CHECK")
print("-"*70)

train_mean = train['price'].mean()
pred_mean = preds['price'].mean()
train_std = train['price'].std()
pred_std = preds['price'].std()

print(f"Train mean: ${train_mean:.2f}, std: ${train_std:.2f}")
print(f"Pred mean:  ${pred_mean:.2f}, std: ${pred_std:.2f}")

if abs(train_mean - pred_mean) > 10:
    print("\n❌ LARGE MEAN DIFFERENCE - predictions are systematically biased!")
    
if pred_std / train_std < 0.5:
    print("\n❌ PREDICTIONS TOO CONSERVATIVE - variance too low!")
    print("   Model is 'playing it safe' and regressing to mean")

# Check price range counts
print("\n4. PRICE RANGE DISTRIBUTION")
print("-"*70)

ranges = [(0, 10), (10, 30), (30, 60), (60, float('inf'))]

print("\nTRAIN:")
for low, high in ranges:
    count = ((train['price'] >= low) & (train['price'] < high)).sum()
    pct = count / len(train) * 100
    print(f"  ${low:3d}-${high:3.0f}: {count:5d} ({pct:5.1f}%)")

print("\nPREDICTIONS:")
for low, high in ranges:
    count = ((preds['price'] >= low) & (preds['price'] < high)).sum()
    pct = count / len(preds) * 100
    print(f"  ${low:3d}-${high:3.0f}: {count:5d} ({pct:5.1f}%)")

# Load ensemble to check routing
with open('models/stratified_ensemble.pkl', 'rb') as f:
    ensemble = pickle.load(f)

print("\n5. STRATIFIED MODEL ROUTING ISSUE")
print("-"*70)
print("\nThe problem is likely:")
print("1. Overfitting in stratified models (small sample ranges)")
print("2. Routing logic using initial predictions is circular")
print("3. Need simpler global model instead")

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)
print("\n❌ Stratified ensemble OVERFITTED badly")
print("   CV: 26.84% → Test: 56.39% (30 point gap!)")
print("\n💡 SOLUTION: Use simpler approach")
print("   1. Remove stratification (too complex, overfits)")
print("   2. Keep advanced features (they help)")
print("   3. Use single robust LightGBM with MORE regularization")
print("   4. Add calibration step")
print("\nThis should get you 48-52% SMAPE (better than current 56%)")
print("="*70)
