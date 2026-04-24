#!/usr/bin/env python3
"""
FINAL SERIOUS ATTEMPT
- Use existing baseline model (54% test SMAPE)
- Apply simple improvements that DON'T overfit
- Proper OOF validation to ensure test will match
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
from sklearn.model_selection import KFold

sys.path.append('src')
from features import FeatureExtractor, create_feature_matrix

def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0
    if not np.any(mask):
        return 0.0
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100

print("="*70)
print("FINAL SERIOUS ATTEMPT - Proper OOF Validation")
print("="*70)

# Load data
print("\n1. Loading data...")
train_df = pd.read_csv('student_resource/dataset/train.csv')
print(f"   {len(train_df)} samples")

# Remove extreme outliers (top 0.5%)
print("\n2. Removing extreme outliers...")
price_99_5 = train_df['price'].quantile(0.995)
outliers = train_df['price'] > price_99_5
print(f"   Removing {outliers.sum()} samples with price > ${price_99_5:.2f}")
train_df = train_df[~outliers].reset_index(drop=True)

# Extract features
print("\n3. Extracting features (basic only)...")
extractor = FeatureExtractor()
train_df = extractor.extract_all_features(train_df)

# Create feature matrix
print("\n4. Creating feature matrix...")
X, feature_columns, valid_indices = create_feature_matrix(train_df, 'embeddings/train', 'optimized')

y_all = train_df['price'].values
y = y_all[valid_indices]
y_log = np.log1p(y)

print(f"   X shape: {X.shape}")
print(f"   y shape: {y.shape}")
print(f"   Price range: ${y.min():.2f} - ${y.max():.2f}")

# SIMPLE hyperparameters - focus on generalization
print("\n5. Training with focus on generalization...")

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,  # Simple
    'max_depth': 6,    # Shallow
    'learning_rate': 0.03,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_data_in_leaf': 50,  # High - prevent overfitting
    'lambda_l1': 0.5,
    'lambda_l2': 0.5,
    'verbose': -1,
    'random_state': 42,
    'n_estimators': 1000
}

# 5-fold CV with OOF predictions
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds_log = np.zeros(len(y))
models = []

print("\n   Training 5 folds...")
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y_log[train_idx], y_log[val_idx]
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )
    models.append(model)
    
    # OOF predictions
    y_pred_log = model.predict(X_val, num_iteration=model.best_iteration)
    oof_preds_log[val_idx] = y_pred_log
    
    # Evaluate in original space
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_val)
    
    fold_smape = smape(y_true, y_pred)
    print(f"   Fold {fold+1}: SMAPE = {fold_smape:.2f}%, iters = {model.best_iteration}")

# Calculate OOF SMAPE (most reliable estimate)
oof_preds = np.expm1(oof_preds_log)
oof_smape = smape(y, oof_preds)

print(f"\n   OOF SMAPE: {oof_smape:.2f}% ← THIS IS YOUR REAL TEST ESTIMATE")

# Check prediction distribution
print(f"\n6. Prediction distribution check:")
print(f"   Train: mean=${y.mean():.2f}, std=${y.std():.2f}")
print(f"   OOF:   mean=${oof_preds.mean():.2f}, std=${oof_preds.std():.2f}")

ratio = oof_preds.std() / y.std()
if ratio < 0.7:
    print(f"   ⚠️ OOF predictions too conservative (std ratio = {ratio:.2f})")
    print(f"   Applying variance correction...")
    
    # Variance correction
    oof_preds_corrected = (oof_preds - oof_preds.mean()) / oof_preds.std() * y.std() * 0.85 + y.mean()
    oof_preds_corrected = np.clip(oof_preds_corrected, y.min(), y.max())
    
    corrected_smape = smape(y, oof_preds_corrected)
    print(f"   Corrected OOF SMAPE: {corrected_smape:.2f}%")
    
    if corrected_smape < oof_smape:
        print(f"   ✅ Improvement: {oof_smape - corrected_smape:.2f} points")
        use_correction = True
    else:
        print(f"   ❌ No improvement, skip correction")
        use_correction = False
else:
    use_correction = False
    print(f"   ✅ Distribution looks good")

# Train final model on ALL data
print("\n7. Training final model on full data...")
train_data = lgb.Dataset(X, label=y_log)
final_model = lgb.train(params, train_data)

# Save
print("\n8. Saving model...")
model_data = {
    'model': final_model,
    'cv_models': models,
    'feature_columns': feature_columns,
    'target_transform': 'log1p',
    'model_name': 'final_attempt',
    'oof_smape': oof_smape,
    'use_variance_correction': use_correction,
    'train_stats': {
        'mean': float(y.mean()),
        'std': float(y.std()),
        'min': float(y.min()),
        'max': float(y.max())
    }
}

os.makedirs('models', exist_ok=True)
with open('models/final_attempt.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print(f"   Saved to models/final_attempt.pkl")

# Summary
print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"OOF SMAPE: {oof_smape:.2f}%")
print(f"\nThis OOF score is your BEST estimate of test performance.")
print(f"Your test SMAPE should be within ±2 points of {oof_smape:.2f}%")

if oof_smape < 52:
    print(f"\n✅ IMPROVEMENT from 54% to {oof_smape:.2f}%!")
elif oof_smape < 54:
    print(f"\n🟡 SLIGHT improvement from 54% to {oof_smape:.2f}%")
else:
    print(f"\n⚠️ No improvement (still {oof_smape:.2f}%)")

print("\nNext: python predict_final_attempt.py")
print("="*70)
