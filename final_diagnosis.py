import pandas as pd
import numpy as np
import pickle

print("="*70)
print("FINAL DIAGNOSIS: WHY 54% SMAPE?")
print("="*70)

# Load your latest model
try:
    with open('models/baseline_lgb.pkl', 'rb') as f:
        model_data = pickle.load(f)
    print("\n✅ Model loaded successfully")
    print(f"CV SMAPE reported during training: {model_data['cv_scores']['smape']:.2f}%")
except Exception as e:
    print(f"\n❌ Could not load model: {e}")
    exit(1)

# Load train data
train = pd.read_csv('student_resource/dataset/train.csv')

# Make predictions on training data itself
print("\n" + "="*70)
print("TEST 1: Predictions on TRAINING data")
print("="*70)

from src.features import FeatureExtractor, create_feature_matrix

extractor = FeatureExtractor()
features_df = extractor.extract_all_features(train)
X, _, valid_indices = create_feature_matrix(features_df, 'embeddings/train', 'optimized')

# Get aligned targets
y_true = train['price'].values[valid_indices]

# Predict
y_pred_log = model_data['model'].predict(X)
y_pred = np.expm1(y_pred_log)

# Calculate SMAPE
def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100

train_smape = smape(y_true, y_pred)
print(f"\nSMAPE on training data: {train_smape:.2f}%")
print(f"CV SMAPE (from training): {model_data['cv_scores']['smape']:.2f}%")

if abs(train_smape - model_data['cv_scores']['smape']) > 5:
    print("\n⚠️ BIG MISMATCH between train SMAPE and CV SMAPE!")
    print("   This suggests overfitting or evaluation bug")

# Analyze prediction distribution
print("\n" + "="*70)
print("TEST 2: Prediction Distribution Analysis")
print("="*70)
print(f"\nTrue prices:")
print(f"  Mean: ${np.mean(y_true):.2f}, Median: ${np.median(y_true):.2f}")
print(f"  Min: ${np.min(y_true):.2f}, Max: ${np.max(y_true):.2f}")

print(f"\nPredicted prices:")
print(f"  Mean: ${np.mean(y_pred):.2f}, Median: ${np.median(y_pred):.2f}")  
print(f"  Min: ${np.min(y_pred):.2f}, Max: ${np.max(y_pred):.2f}")

# Check if predictions are too conservative
pred_std = np.std(y_pred)
true_std = np.std(y_true)
print(f"\nVariance comparison:")
print(f"  True std: ${true_std:.2f}")
print(f"  Pred std: ${pred_std:.2f}")
print(f"  Ratio: {pred_std/true_std:.2f}")

if pred_std / true_std < 0.5:
    print("\n❌ PROBLEM FOUND: Predictions are TOO CONSERVATIVE!")
    print("   Model is 'hedging' and predicting values close to mean")
    print("   This is classic REGRESSION TO THE MEAN problem")
    print("\n💡 SOLUTION:")
    print("   1. Reduce regularization (lambda_l1, lambda_l2)")
    print("   2. Increase model complexity (more leaves, deeper trees)")
    print("   3. Try different loss functions")

# Analyze by price range
print("\n" + "="*70)
print("TEST 3: SMAPE by Price Range")
print("="*70)

price_ranges = [
    (0, 10, "Under $10"),
    (10, 30, "$10-$30"),
    (30, 60, "$30-$60"),
    (60, 100, "$60-$100"),
    (100, float('inf'), "Over $100")
]

for low, high, label in price_ranges:
    mask = (y_true >= low) & (y_true < high)
    if mask.sum() > 0:
        range_smape = smape(y_true[mask], y_pred[mask])
        print(f"{label:15s}: {range_smape:6.2f}% (n={mask.sum():5d})")

# Check for systematic bias
print("\n" + "="*70)
print("TEST 4: Systematic Bias Check")
print("="*70)

residuals = y_pred - y_true
print(f"Mean residual: ${np.mean(residuals):.2f}")
print(f"Median residual: ${np.median(residuals):.2f}")

if abs(np.mean(residuals)) > 5:
    if np.mean(residuals) > 0:
        print("\n⚠️ Model systematically OVERPREDICTS prices")
    else:
        print("\n⚠️ Model systematically UNDERPREDICTS prices")
    print("💡 Add a bias correction term in post-processing")

# Check worst predictions
print("\n" + "="*70)
print("TEST 5: Worst Predictions")
print("="*70)

# Calculate per-sample SMAPE
per_sample_smape = np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) / 2) * 100

worst_indices = np.argsort(per_sample_smape)[-5:]
print("\nTop 5 worst predictions:")
for i, idx in enumerate(worst_indices[::-1], 1):
    true_price = y_true[idx]
    pred_price = y_pred[idx]
    sample_smape = per_sample_smape[idx]
    sample_id = train.iloc[valid_indices[idx]]['sample_id']
    text = train.iloc[valid_indices[idx]]['catalog_content'][:80]
    print(f"\n{i}. Sample {sample_id}")
    print(f"   True: ${true_price:.2f}, Pred: ${pred_price:.2f}, SMAPE: {sample_smape:.1f}%")
    print(f"   Text: {text}...")

# FINAL RECOMMENDATIONS
print("\n" + "="*70)
print("FINAL RECOMMENDATIONS")
print("="*70)

if train_smape > 50:
    print("\n🔴 YOUR MODEL IS UNDERFITTING")
    print("\n💡 Actions to take:")
    print("   1. REDUCE regularization:")
    print("      - Set lambda_l1=0.01, lambda_l2=0.01 (instead of 1.0)")
    print("      - Reduce min_data_in_leaf to 20 (instead of 50)")
    print("   2. INCREASE model capacity:")
    print("      - Set num_leaves=127")
    print("      - Remove max_depth limit")
    print("   3. Train LONGER:")
    print("      - Set n_estimators=2000")
    print("      - Increase early_stopping to 200")
else:
    print("\n🟡 YOUR MODEL IS REASONABLE BUT CAN BE BETTER")
    print("\n💡 Actions to take:")
    print("   1. Try ensemble of models")
    print("   2. Add price-range-specific post-processing")
    print("   3. Remove extreme outliers from training")

print("\n" + "="*70)
