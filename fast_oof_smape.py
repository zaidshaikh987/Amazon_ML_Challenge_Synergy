import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
import hnswlib
from scipy import interpolate


def smape(y_true, y_pred):
    return np.mean(200 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))


print("="*70)
print("FAST OOF SMAPE: LGBM, k-NN (LOO), and Blends")
print("="*70)

# 1) Load train embeddings and target
X_img = np.load("embeddings/train/image_embeddings.npy")
X_txt = np.load("embeddings/train/text_embeddings.npy")
X = np.hstack([X_img, X_txt]).astype('float32')
train_df = pd.read_csv("student_resource/dataset/train.csv")
y = train_df['price'].values.astype('float32')
N, D = X.shape

print(f"Train: {N} samples, dim={D}")

# 2) k-NN leave-one-out predictions via single HNSW index (fast, approx OOF)
print("\n[1/3] Building HNSW index for leave-one-out k-NN...")
hnsw = hnswlib.Index(space='l2', dim=D)
hnsw.init_index(max_elements=N, ef_construction=100, M=16)
hnsw.add_items(X, np.arange(N))
hnsw.set_ef(100)

K = 5
labels, dists = hnsw.knn_query(X, k=K+1)  # +1 to allow self

neighbor_ids = []
for i in range(N):
    li = labels[i]
    # drop self if present
    if li[0] == i:
        neighbor_ids.append(li[1:K+1])
    else:
        neighbor_ids.append(li[:K])
neighbor_ids = np.array(neighbor_ids)

knn_pred = np.median(y[neighbor_ids], axis=1)
knn_smape = smape(y, knn_pred)
print(f"k-NN (LOO) SMAPE: {knn_smape:.2f}%")

# 3) 5-fold LightGBM OOF
print("\n[2/3] Training LightGBM 5-fold OOF...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
lgb_oof = np.zeros(N, dtype='float32')

for fold, (tr, va) in enumerate(kf.split(X), 1):
    X_tr, X_va = X[tr], X[va]
    y_tr, y_va = y[tr], y[va]

    model = LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_tr, np.log1p(y_tr))
    lgb_oof[va] = np.expm1(model.predict(X_va))
    print(f"   Fold {fold}/5 done")

lgb_smape = smape(y, lgb_oof)
print(f"LightGBM OOF SMAPE: {lgb_smape:.2f}%")

# 4) Blends OOF
print("\n[3/3] Computing blends and calibrated blends...")
ensembles = [
    ("Blend 50/50", 0.5, 0.5),
    ("Blend 60/40 (favor LGB)", 0.6, 0.4),
    ("Blend 70/30 (favor LGB)", 0.7, 0.3),
]

# Calibrator helpers
p1, p99 = np.percentile(y, [1, 99])
quantiles = np.linspace(0, 1, 100)
train_quantiles = np.percentile(y, quantiles * 100)

def calibrate(preds, alpha=0.7):
    # clip
    clipped = np.clip(preds, p1, p99)
    # quantile map
    pred_q = np.percentile(preds, quantiles * 100)
    cal = interpolate.interp1d(pred_q, train_quantiles, kind='linear', bounds_error=False,
                               fill_value=(train_quantiles[0], train_quantiles[-1]))
    qmapped = cal(preds)
    return alpha * clipped + (1 - alpha) * qmapped

best_baseline = min(lgb_smape, knn_smape)

rows = []
rows.append(("k-NN (LOO)", knn_smape))
rows.append(("LightGBM OOF", lgb_smape))

for name, w_lgb, w_knn in ensembles:
    blend = w_lgb * lgb_oof + w_knn * knn_pred
    s1 = smape(y, blend)
    cal_blend = calibrate(blend, alpha=0.7)
    s2 = smape(y, cal_blend)
    rows.append((name, s1))
    rows.append((name + " + Calibration", s2))

# Report
print("\n" + "="*70)
print(f"{'MODEL':40s} SMAPE")
print("-"*70)
for name, s in sorted(rows, key=lambda x: x[1]):
    print(f"{name:40s} {s:5.2f}%")
print("="*70)
