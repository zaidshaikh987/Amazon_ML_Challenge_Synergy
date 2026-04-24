#!/usr/bin/env python3
"""
k-NN Price Predictor with HNSW
Fast, simple, and often effective for pricing tasks!
"""

import numpy as np
import pandas as pd
import hnswlib
from sklearn.model_selection import KFold

def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0
    if not np.any(mask):
        return 0.0
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100

print("="*70)
print("k-NN PRICE PREDICTOR WITH HNSW")
print("="*70)

# Load train embeddings
print("\n1. Loading train embeddings...")
X_img = np.load("embeddings/train/image_embeddings.npy")
X_txt = np.load("embeddings/train/text_embeddings.npy")
X = np.hstack([X_img, X_txt]).astype('float32')

# Load train prices
train_df = pd.read_csv("student_resource/dataset/train.csv")
y = train_df['price'].values

print(f"   Train embeddings: {X.shape}")
print(f"   Train prices: {y.shape}")

# OOF evaluation to get REAL performance estimate
print("\n2. Running 5-fold OOF evaluation...")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(y))

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\n   Fold {fold+1}/5...")
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train = y[train_idx]
    
    # Build HNSW index on training data
    dim = X.shape[1]
    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=len(train_idx), ef_construction=200, M=32)
    p.add_items(X_train, np.arange(len(train_idx)))
    p.set_ef(100)
    
    # Query validation data
    labels, distances = p.knn_query(X_val, k=10)  # top-10 neighbors
    
    # Try different k values
    for k in [5, 10]:
        preds_k = np.median(y_train[labels[:, :k]], axis=1)
        smape_k = smape(y[val_idx], preds_k)
        print(f"      k={k:2d}: SMAPE = {smape_k:.2f}%")
    
    # Use k=10 for OOF
    oof_preds[val_idx] = np.median(y_train[labels[:, :10]], axis=1)

# Calculate OOF SMAPE
oof_smape = smape(y, oof_preds)
print(f"\n   OOF SMAPE (k=10): {oof_smape:.2f}%")

# Check distribution
print(f"\n3. Distribution check:")
print(f"   Train: mean=${y.mean():.2f}, std=${y.std():.2f}")
print(f"   OOF:   mean=${oof_preds.mean():.2f}, std=${oof_preds.std():.2f}")

# Build final index on ALL training data
print("\n4. Building final k-NN index on all training data...")
dim = X.shape[1]
p = hnswlib.Index(space='l2', dim=dim)
p.init_index(max_elements=X.shape[0], ef_construction=200, M=32)
p.add_items(X, np.arange(X.shape[0]))
p.set_ef(100)

# Load test embeddings
print("\n5. Loading test embeddings...")
X_test_img = np.load("embeddings/test/image_embeddings.npy")
X_test_txt = np.load("embeddings/test/text_embeddings.npy")
X_test = np.hstack([X_test_img, X_test_txt]).astype('float32')

print(f"   Test embeddings: {X_test.shape}")

# Query test set
print("\n6. Generating k-NN predictions for test set...")
labels, distances = p.knn_query(X_test, k=10)

# Median price of neighbors
knn_preds = np.median(y[labels], axis=1)

# Load test sample IDs
test_df = pd.read_csv("student_resource/dataset/test.csv")

# Save k-NN predictions
knn_df = pd.DataFrame({
    'sample_id': test_df['sample_id'],
    'price': knn_preds
})

import os
os.makedirs('submissions', exist_ok=True)
knn_df.to_csv("submissions/knn_predictions.csv", index=False)

print(f"\n7. Saved to submissions/knn_predictions.csv")
print(f"   Test predictions: mean=${knn_preds.mean():.2f}, std=${knn_preds.std():.2f}")

# Summary
print("\n" + "="*70)
print("k-NN RESULTS")
print("="*70)
print(f"OOF SMAPE: {oof_smape:.2f}%")

if oof_smape < 54:
    print(f"\n✅ k-NN is BETTER than LightGBM (54%)!")
    print(f"   Use k-NN alone or blend with LightGBM")
else:
    print(f"\n⚠️ k-NN ({oof_smape:.2f}%) is worse than LightGBM (54%)")
    print(f"   But blending might still help!")

print("\nNext: Blend k-NN with your LightGBM predictions")
print("="*70)
