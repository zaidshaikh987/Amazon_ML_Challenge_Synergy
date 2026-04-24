import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold

def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0
    if not np.any(mask):
        return 0.0
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100

print("="*70)
print("k-NN CROSS-VALIDATION (ESTIMATING TEST SMAPE)")
print("="*70)

# Load train data
print("\n1. Loading train data...")
X_img = np.load("embeddings/train/image_embeddings.npy")
X_txt = np.load("embeddings/train/text_embeddings.npy")
X = np.hstack([X_img, X_txt])

train_df = pd.read_csv("student_resource/dataset/train.csv")
y = train_df['price'].values

print(f"   Data: {X.shape}, prices: {y.shape}")

# 5-fold CV
print("\n2. Running 5-fold cross-validation...")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Build k-NN on train fold
    knn = NearestNeighbors(n_neighbors=10, metric='cosine')
    knn.fit(X_train)
    
    # Predict on validation fold
    distances, indices = knn.kneighbors(X_val)
    y_pred = np.median(y_train[indices], axis=1)
    
    # Calculate SMAPE
    fold_smape = smape(y_val, y_pred)
    cv_scores.append(fold_smape)
    
    print(f"   Fold {fold+1}: SMAPE = {fold_smape:.2f}%")

mean_smape = np.mean(cv_scores)
std_smape = np.std(cv_scores)

print(f"\n   Mean CV SMAPE: {mean_smape:.2f}% ± {std_smape:.2f}%")

# Compare to your LightGBM
print("\n" + "="*70)
print("COMPARISON")
print("="*70)
print(f"k-NN CV SMAPE:      {mean_smape:.2f}%")
print(f"LightGBM test SMAPE: 54.00%")

if mean_smape < 54:
    print(f"\n✅ k-NN is {54 - mean_smape:.2f} points BETTER!")
    print("   Submit knn_predictions.csv")
elif mean_smape < 56:
    print(f"\n🟡 k-NN is similar ({mean_smape:.2f}% vs 54%)")
    print("   Try blending k-NN + LightGBM")
else:
    print(f"\n❌ k-NN is worse ({mean_smape:.2f}% vs 54%)")
    print("   Stick with LightGBM")

print("="*70)
