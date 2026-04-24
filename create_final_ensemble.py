import pandas as pd
import numpy as np
from scipy.optimize import minimize
import sys
sys.path.append('src')
from calibration import PredictionCalibrator

print("="*70)
print("COMPREHENSIVE ENSEMBLE: LightGBM + k-NN + Neural Fusion + Calibration")
print("="*70)

# ========== 1. Load all predictions ==========
print("\n1. Loading model predictions...")

# Load test predictions
try:
    lgb = pd.read_csv("submissions/test_out.csv")
    print("   ✓ LightGBM predictions loaded")
except:
    print("   ✗ LightGBM predictions not found")
    lgb = None

try:
    knn = pd.read_csv("submissions/knn_predictions.csv")
    print("   ✓ k-NN predictions loaded")
except:
    print("   ✗ k-NN predictions not found")
    knn = None

try:
    neural = pd.read_csv("submissions/neural_fusion_predictions.csv")
    print("   ✓ Neural Fusion predictions loaded")
except:
    print("   ✗ Neural Fusion predictions not found (run train_neural_fusion.py first)")
    neural = None

# Load train data for calibration
train = pd.read_csv("student_resource/dataset/train.csv")
test = pd.read_csv("student_resource/dataset/test.csv")

# ========== 2. Merge predictions ==========
print("\n2. Merging predictions...")

merged = test[['sample_id']].copy()

if lgb is not None:
    merged = merged.merge(lgb, on='sample_id', how='left')
    merged.rename(columns={'price': 'price_lgb'}, inplace=True)

if knn is not None:
    merged = merged.merge(knn[['sample_id', 'price']], on='sample_id', how='left')
    merged.rename(columns={'price': 'price_knn'}, inplace=True)

if neural is not None:
    merged = merged.merge(neural, on='sample_id', how='left')
    merged.rename(columns={'price': 'price_neural'}, inplace=True)

print(f"   Merged shape: {merged.shape}")
print(f"   Columns: {merged.columns.tolist()}")

# ========== 3. Statistics ==========
print("\n3. Model statistics:")

if 'price_lgb' in merged.columns:
    print(f"   LightGBM:     mean=${merged['price_lgb'].mean():.2f}, std=${merged['price_lgb'].std():.2f}")
if 'price_knn' in merged.columns:
    print(f"   k-NN:         mean=${merged['price_knn'].mean():.2f}, std=${merged['price_knn'].std():.2f}")
if 'price_neural' in merged.columns:
    print(f"   Neural:       mean=${merged['price_neural'].mean():.2f}, std=${merged['price_neural'].std():.2f}")

print(f"   Train:        mean=${train['price'].mean():.2f}, std=${train['price'].std():.2f}")

# ========== 4. Create ensemble combinations ==========
print("\n4. Creating ensemble combinations...")

calibrator = PredictionCalibrator(train['price'].values)

ensembles = {}

# Simple blends
if 'price_lgb' in merged.columns and 'price_knn' in merged.columns:
    ensembles['lgb_knn_50_50'] = 0.5 * merged['price_lgb'] + 0.5 * merged['price_knn']
    ensembles['lgb_knn_60_40'] = 0.6 * merged['price_lgb'] + 0.4 * merged['price_knn']
    ensembles['lgb_knn_70_30'] = 0.7 * merged['price_lgb'] + 0.3 * merged['price_knn']

# Three-way blends (if neural available)
if 'price_lgb' in merged.columns and 'price_knn' in merged.columns and 'price_neural' in merged.columns:
    # Equal weight
    ensembles['all_equal'] = (merged['price_lgb'] + merged['price_knn'] + merged['price_neural']) / 3
    
    # Weighted combinations
    ensembles['all_50_30_20'] = 0.5 * merged['price_lgb'] + 0.3 * merged['price_knn'] + 0.2 * merged['price_neural']
    ensembles['all_40_40_20'] = 0.4 * merged['price_lgb'] + 0.4 * merged['price_knn'] + 0.2 * merged['price_neural']
    ensembles['all_45_35_20'] = 0.45 * merged['price_lgb'] + 0.35 * merged['price_knn'] + 0.2 * merged['price_neural']

# Calibrated versions
if 'price_lgb' in merged.columns:
    ensembles['lgb_calibrated_clip'] = calibrator.clip_predictions(merged['price_lgb'].values)
    ensembles['lgb_calibrated_quantile'] = calibrator.quantile_calibrate(merged['price_lgb'].values)
    ensembles['lgb_calibrated_conservative'] = calibrator.calibrate_conservative(merged['price_lgb'].values, alpha=0.7)

if 'price_knn' in merged.columns:
    ensembles['knn_calibrated'] = calibrator.calibrate_conservative(merged['price_knn'].values, alpha=0.7)

# Calibrated blends
if 'price_lgb' in merged.columns and 'price_knn' in merged.columns:
    blend_50_50 = 0.5 * merged['price_lgb'] + 0.5 * merged['price_knn']
    ensembles['blend_50_50_calibrated'] = calibrator.calibrate_conservative(blend_50_50.values, alpha=0.7)
    
    blend_60_40 = 0.6 * merged['price_lgb'] + 0.4 * merged['price_knn']
    ensembles['blend_60_40_calibrated'] = calibrator.calibrate_conservative(blend_60_40.values, alpha=0.7)

# ========== 5. Save all ensembles ==========
print(f"\n5. Saving {len(ensembles)} ensemble combinations...")

for name, predictions in ensembles.items():
    result = pd.DataFrame({
        'sample_id': merged['sample_id'],
        'price': predictions
    })
    
    filename = f"submissions/ensemble_{name}.csv"
    result.to_csv(filename, index=False)
    
    print(f"   ✓ {name:40s} | mean=${predictions.mean():.2f}, std=${predictions.std():.2f}")

# ========== 6. Recommendations ==========
print("\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)

print("\n📊 Top submissions to try (in order):")
print("\n1. ensemble_blend_50_50_calibrated.csv")
print("   - Balanced LightGBM + k-NN blend with calibration")
print("   - Reduces extreme errors")
print("\n2. ensemble_all_50_30_20.csv (if neural available)")
print("   - Three-model blend: 50% LGB, 30% kNN, 20% Neural")
print("   - Captures diverse patterns")
print("\n3. ensemble_lgb_calibrated_conservative.csv")
print("   - LightGBM with conservative calibration")
print("   - Safe single-model submission")
print("\n4. ensemble_lgb_knn_60_40.csv")
print("   - Favor LightGBM slightly over k-NN")
print("   - No calibration, good baseline")

print("\n💡 Expected improvements:")
print("   - Calibration: ~0.5-1.5% SMAPE reduction")
print("   - Multi-model blending: ~1-2% SMAPE reduction")
print("   - Combined: Target 51-53% SMAPE")

print("\n" + "="*70)
print("NEXT STEPS")
print("="*70)
print("\n1. Submit ensemble_blend_50_50_calibrated.csv first")
print("2. If neural fusion is trained, try ensemble_all_50_30_20.csv")
print("3. Compare results and iterate on blend weights")
print("4. Consider adding aggregate features (run src/aggregate_features.py)")
print("\n" + "="*70)
