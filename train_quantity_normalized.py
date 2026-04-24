#!/usr/bin/env python3
"""
CRITICAL FIX: Train on PER-UNIT prices, not total prices.
This could be the 13-point gap to 41% SMAPE!
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
from sklearn.model_selection import KFold
import re

sys.path.append('src')
from features import FeatureExtractor, create_feature_matrix

def smape(y_true, y_pred):
    """Calculate SMAPE."""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0
    if not np.any(mask):
        return 0.0
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100

def extract_quantity_robust(text):
    """ROBUST quantity extraction - handles many patterns."""
    if pd.isna(text):
        return 1.0
    
    text = str(text).lower()
    
    # Special words
    if 'dozen' in text:
        return 12.0
    if ' pair' in text or 'pairs' in text:
        return 2.0
    
    # Patterns (ordered by specificity)
    patterns = [
        r'pack\s+of\s+(\d+)',
        r'(\d+)\s*pack',
        r'(\d+)\s*-\s*pack',
        r'(\d+)pk',
        r'(\d+)\s+pcs',
        r'(\d+)\s+pieces',
        r'(\d+)\s+count',
        r'(\d+)\s+ct',
        r'set\s+of\s+(\d+)',
        r'(\d+)\s+set',
        r'box\s+of\s+(\d+)',
        r'case\s+of\s+(\d+)',
        r'(\d+)\s*x\s*\d+',  # "2 x 500ml"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                qty = float(match.group(1))
                if 1 <= qty <= 1000:  # Reasonable range
                    return qty
            except:
                pass
    
    return 1.0  # Default: single item

def main():
    print("="*70)
    print("QUANTITY-NORMALIZED TRAINING")
    print("Critical Fix: Train on per-unit prices!")
    print("="*70)
    
    # Load data
    print("\n1. Loading data...")
    train_df = pd.read_csv('student_resource/dataset/train.csv')
    print(f"   Loaded {len(train_df)} samples")
    
    # Extract quantities
    print("\n2. Extracting quantities...")
    train_df['quantity'] = train_df['catalog_content'].apply(extract_quantity_robust)
    
    qty_counts = train_df['quantity'].value_counts().head(10)
    print(f"\n   Top quantities detected:")
    for qty, count in qty_counts.items():
        print(f"     {qty:.0f}x: {count} samples")
    
    # Calculate per-unit prices
    print("\n3. Computing per-unit prices...")
    train_df['price_per_unit'] = train_df['price'] / train_df['quantity']
    
    print(f"\n   Original price range: ${train_df['price'].min():.2f} - ${train_df['price'].max():.2f}")
    print(f"   Per-unit price range: ${train_df['price_per_unit'].min():.2f} - ${train_df['price_per_unit'].max():.2f}")
    
    # Extract features
    print("\n4. Extracting features...")
    extractor = FeatureExtractor()
    train_df = extractor.extract_all_features(train_df)
    
    # Create feature matrix
    print("\n5. Creating feature matrix...")
    X, feature_columns, valid_indices = create_feature_matrix(train_df, 'embeddings/train', 'optimized')
    
    # Get per-unit prices and quantities
    y_per_unit = train_df['price_per_unit'].values[valid_indices]
    quantities = train_df['quantity'].values[valid_indices]
    
    # Log transform per-unit prices
    y_per_unit_log = np.log1p(y_per_unit)
    
    print(f"\n   Feature matrix: {X.shape}")
    print(f"   Training on PER-UNIT prices")
    
    # Simple LightGBM - let it learn properly now
    print("\n6. Training model on per-unit prices...")
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'max_depth': 8,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 20,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'verbose': -1,
        'random_state': 42,
        'n_estimators': 1000
    }
    
    # 5-fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    oof_preds_per_unit_log = np.zeros(len(y_per_unit))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n  Fold {fold+1}/5...")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_per_unit_log[train_idx], y_per_unit_log[val_idx]
        qty_val = quantities[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        # Predict per-unit prices
        y_pred_per_unit_log = model.predict(X_val, num_iteration=model.best_iteration)
        oof_preds_per_unit_log[val_idx] = y_pred_per_unit_log
        
        # Convert to per-unit prices
        y_pred_per_unit = np.expm1(y_pred_per_unit_log)
        y_true_per_unit = np.expm1(y_val)
        
        # Multiply by quantity to get TOTAL prices for evaluation
        y_pred_total = y_pred_per_unit * qty_val
        y_true_total = y_true_per_unit * qty_val
        
        # Calculate SMAPE on TOTAL prices (what competition evaluates)
        fold_smape = smape(y_true_total, y_pred_total)
        cv_scores.append(fold_smape)
        
        print(f"    SMAPE: {fold_smape:.2f}%")
    
    # Calculate OOF SMAPE
    oof_per_unit = np.expm1(oof_preds_per_unit_log)
    oof_total = oof_per_unit * quantities
    true_total = y_per_unit * quantities
    oof_smape = smape(true_total, oof_total)
    
    print(f"\n  Mean CV SMAPE: {np.mean(cv_scores):.2f}% ± {np.std(cv_scores):.2f}%")
    print(f"  OOF SMAPE:     {oof_smape:.2f}%")
    
    # Train final model
    print("\n7. Training final model...")
    train_data = lgb.Dataset(X, label=y_per_unit_log)
    final_model = lgb.train(params, train_data)
    
    # Save model
    print("\n8. Saving model...")
    model_data = {
        'model': final_model,
        'feature_columns': feature_columns,
        'target_transform': 'log1p',
        'model_name': 'quantity_normalized',
        'oof_smape': oof_smape,
        'cv_smapes': cv_scores,
        'predicts_per_unit': True  # IMPORTANT FLAG
    }
    
    os.makedirs('models', exist_ok=True)
    with open('models/quantity_normalized.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"   Saved to models/quantity_normalized.pkl")
    
    # Summary
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"OOF SMAPE: {oof_smape:.2f}%")
    
    if oof_smape < 50:
        print(f"\n✅ IMPROVEMENT! Down from 54% to {oof_smape:.2f}%")
    else:
        print(f"\n⚠️ SMAPE: {oof_smape:.2f}% (no improvement)")
    
    print("\nUse: python predict_quantity_normalized.py")
    print("="*70)

if __name__ == "__main__":
    main()
