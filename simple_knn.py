import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

print("="*70)
print("SIMPLE k-NN PRICE PREDICTOR")
print("="*70)

# Load train data
print("\n1. Loading train data...")
X_img = np.load("embeddings/train/image_embeddings.npy")
X_txt = np.load("embeddings/train/text_embeddings.npy")
X_train = np.hstack([X_img, X_txt])

train_df = pd.read_csv("student_resource/dataset/train.csv")
y_train = train_df['price'].values

print(f"   Train: {X_train.shape}, prices: {y_train.shape}")

# Load test data
print("\n2. Loading test data...")
X_test_img = np.load("embeddings/test/image_embeddings.npy")
X_test_txt = np.load("embeddings/test/text_embeddings.npy")
X_test = np.hstack([X_test_img, X_test_txt])

test_df = pd.read_csv("student_resource/dataset/test.csv")

print(f"   Test: {X_test.shape}")

# Build k-NN model
print("\n3. Building k-NN model (k=10)...")
knn = NearestNeighbors(n_neighbors=10, metric='cosine', n_jobs=-1)
knn.fit(X_train)

# Find neighbors
print("\n4. Finding neighbors for test set...")
distances, indices = knn.kneighbors(X_test)

# Predict: median price of neighbors
print("\n5. Computing predictions (median of neighbors)...")
predictions = np.median(y_train[indices], axis=1)

print(f"   Predictions: mean=${predictions.mean():.2f}, std=${predictions.std():.2f}")

# Save
print("\n6. Saving predictions...")
result = pd.DataFrame({
    'sample_id': test_df['sample_id'],
    'price': predictions
})

import os
os.makedirs('submissions', exist_ok=True)
result.to_csv('submissions/knn_predictions.csv', index=False)

print(f"   Saved to submissions/knn_predictions.csv")
print("\n✅ k-NN predictions ready!")
print("   Submit this to see k-NN performance")
print("="*70)
