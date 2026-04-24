"""
Retrain LightGBM with aggregate features.
This should give ~1-1.5% SMAPE improvement.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
import pickle

def smape(y_true, y_pred):
    return np.mean(200 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))

print("="*70)
print("TRAINING LIGHTGBM WITH AGGREGATE FEATURES")
print("="*70)

# Load data with aggregates
print("\n1. Loading data with aggregate features...")
train_agg = pd.read_csv("student_resource/dataset/train_with_aggregates.csv")
test_agg = pd.read_csv("student_resource/dataset/test_with_aggregates.csv")

print(f"   Train: {train_agg.shape}")
print(f"   Test: {test_agg.shape}")

# Load embeddings
X_img_train = np.load("embeddings/train/image_embeddings.npy")
X_txt_train = np.load("embeddings/train/text_embeddings.npy")
X_img_test = np.load("embeddings/test/image_embeddings.npy")
X_txt_test = np.load("embeddings/test/text_embeddings.npy")

# Get aggregate feature columns
agg_cols = [c for c in train_agg.columns if 'price' in c.lower() and c != 'price']
print(f"\n2. Aggregate features ({len(agg_cols)}):")
for col in agg_cols:
    print(f"   • {col}")

# Combine embeddings + aggregate features
X_train_emb = np.hstack([X_img_train, X_txt_train])
X_train_agg = train_agg[agg_cols].values
X_train = np.hstack([X_train_emb, X_train_agg]).astype('float32')

X_test_emb = np.hstack([X_img_test, X_txt_test])
X_test_agg = test_agg[agg_cols].values
X_test = np.hstack([X_test_emb, X_test_agg]).astype('float32')

y_train = train_agg['price'].values

print(f"\n3. Final feature matrix:")
print(f"   Train: {X_train.shape}")
print(f"   Test: {X_test.shape}")
print(f"   Features: {X_train_emb.shape[1]} embeddings + {X_train_agg.shape[1]} aggregates")

# 5-fold CV training
print("\n4. Training with 5-fold cross-validation...")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(y_train))
test_preds = np.zeros((len(X_test), 5))

models = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
    print(f"\n   Fold {fold}/5:")
    
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    
    # Train model
    model = LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    model.fit(
        X_tr, np.log1p(y_tr),
        eval_set=[(X_val, np.log1p(y_val))]
    )
    
    # OOF predictions
    oof_preds[val_idx] = np.expm1(model.predict(X_val))
    
    # Test predictions
    test_preds[:, fold-1] = np.expm1(model.predict(X_test))
    
    # Fold SMAPE
    fold_smape = smape(y_val, oof_preds[val_idx])
    print(f"      Fold SMAPE: {fold_smape:.2f}%")
    
    models.append(model)

# Overall CV SMAPE
cv_smape = smape(y_train, oof_preds)

print("\n" + "="*70)
print("RESULTS")
print("="*70)

print(f"\n✅ Out-of-Fold SMAPE: {cv_smape:.2f}%")
print(f"   Baseline (without aggregates): 54.00%")
print(f"   Improvement: {54.00 - cv_smape:.2f} percentage points")

# Average test predictions
test_preds_avg = test_preds.mean(axis=1)

print(f"\nTest predictions:")
print(f"   Mean: ${test_preds_avg.mean():.2f}")
print(f"   Median: ${np.median(test_preds_avg):.2f}")
print(f"   Std: ${test_preds_avg.std():.2f}")

# Save predictions
print("\n5. Saving predictions...")

test_df = pd.read_csv("student_resource/dataset/test.csv")
submission = pd.DataFrame({
    'sample_id': test_df['sample_id'],
    'price': test_preds_avg
})

submission.to_csv("submissions/lgb_with_aggregates.csv", index=False)
print("   Saved to: submissions/lgb_with_aggregates.csv")

# Save models
with open("models/lgb_with_aggregates.pkl", "wb") as f:
    pickle.dump(models, f)
print("   Models saved to: models/lgb_with_aggregates.pkl")

# Save OOF predictions
oof_df = pd.DataFrame({
    'sample_id': train_agg['sample_id'],
    'price_actual': y_train,
    'price_pred': oof_preds
})
oof_df.to_csv("submissions/lgb_with_aggregates_oof.csv", index=False)
print("   OOF saved to: submissions/lgb_with_aggregates_oof.csv")

print("\n" + "="*70)
print("NEXT STEPS")
print("="*70)

print("\n1. Submit lgb_with_aggregates.csv to competition")
print(f"   Expected SMAPE: ~{cv_smape:.2f}%")

print("\n2. Create ensemble with k-NN:")
print("   Run: python create_ensemble_with_aggregates.py")

print("\n3. If improvement confirmed, this is your new baseline!")

print("\n" + "="*70)
