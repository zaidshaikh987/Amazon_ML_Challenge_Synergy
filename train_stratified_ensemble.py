#!/usr/bin/env python3
"""
Stratified Ensemble: Train separate models for different price ranges.
This can significantly improve SMAPE by specializing models.
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

sys.path.append('src')
from features import FeatureExtractor, create_feature_matrix
from advanced_features import add_advanced_features_to_df

def smape(y_true, y_pred):
    """Calculate SMAPE."""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0
    if not np.any(mask):
        return 0.0
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100

def train_price_range_model(X, y, price_range_name, quick=False):
    """Train a model for a specific price range."""
    
    print(f"\n{'='*60}")
    print(f"Training model for {price_range_name}")
    print(f"Samples: {len(y)}, Price range: ${y.min():.2f} - ${y.max():.2f}")
    print(f"{'='*60}")
    
    # Hyperparameters optimized for each range
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': max(20, len(y) // 100),  # Adaptive
        'lambda_l1': 0.5,
        'lambda_l2': 0.5,
        'verbose': -1,
        'random_state': 42,
        'n_estimators': 300 if quick else 1000
    }
    
    # Log transform
    y_log = np.log1p(y)
    
    # 3-fold CV (smaller for stratified approach)
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = []
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_log[train_idx], y_log[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        models.append(model)
        
        # Predict in original space
        y_pred_log = model.predict(X_val, num_iteration=model.best_iteration)
        y_pred = np.expm1(y_pred_log)
        y_true = np.expm1(y_val)
        
        fold_smape = smape(y_true, y_pred)
        cv_scores.append(fold_smape)
        print(f"  Fold {fold+1}: SMAPE = {fold_smape:.2f}%")
    
    print(f"  Mean SMAPE: {np.mean(cv_scores):.2f}% ± {np.std(cv_scores):.2f}%")
    
    # Train final model on all data for this range
    train_data = lgb.Dataset(X, label=y_log)
    final_model = lgb.train(params, train_data)
    
    return final_model, np.mean(cv_scores)

def main():
    print("="*70)
    print("STRATIFIED ENSEMBLE TRAINING")
    print("="*70)
    
    # Load data
    print("\n1. Loading training data...")
    train_df = pd.read_csv('student_resource/dataset/train.csv')
    print(f"   Loaded {len(train_df)} samples")
    
    # Extract basic features
    print("\n2. Extracting basic features...")
    extractor = FeatureExtractor()
    train_df = extractor.extract_all_features(train_df)
    
    # Extract ADVANCED features
    print("\n3. Extracting advanced features...")
    train_df = add_advanced_features_to_df(train_df)
    
    # Create feature matrix
    print("\n4. Creating feature matrix...")
    X, feature_columns, valid_indices = create_feature_matrix(train_df, 'embeddings/train', 'optimized')
    
    # Get prices
    y_all = train_df['price'].values
    y = y_all[valid_indices]
    
    print(f"   Feature matrix: {X.shape}")
    print(f"   Valid samples: {len(y)}")
    
    # Define price ranges (based on quartiles + outliers)
    price_ranges = [
        (0, 10, "low"),
        (10, 30, "mid_low"),
        (30, 60, "mid_high"),
        (60, float('inf'), "high")
    ]
    
    # Train models for each range
    print("\n5. Training stratified models...")
    stratified_models = {}
    range_smapes = {}
    
    for low, high, name in price_ranges:
        # Filter samples in this range
        mask = (y >= low) & (y < high)
        
        if mask.sum() < 100:  # Skip if too few samples
            print(f"\nSkipping {name} range (only {mask.sum()} samples)")
            continue
        
        X_range = X[mask]
        y_range = y[mask]
        
        # Train model
        model, smape_score = train_price_range_model(X_range, y_range, name, quick=False)
        
        stratified_models[name] = {
            'model': model,
            'price_range': (low, high),
            'cv_smape': smape_score,
            'n_samples': mask.sum()
        }
        range_smapes[name] = smape_score
    
    # Train a fallback global model
    print(f"\n{'='*60}")
    print("Training global fallback model...")
    print(f"{'='*60}")
    
    y_log = np.log1p(y)
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 30,
        'lambda_l1': 0.5,
        'lambda_l2': 0.5,
        'verbose': -1,
        'random_state': 42,
        'n_estimators': 1000
    }
    
    train_data = lgb.Dataset(X, label=y_log)
    global_model = lgb.train(params, train_data)
    
    # Save everything
    print("\n6. Saving stratified ensemble...")
    
    ensemble_data = {
        'stratified_models': stratified_models,
        'global_model': global_model,
        'price_ranges': price_ranges,
        'feature_columns': feature_columns,
        'range_smapes': range_smapes
    }
    
    os.makedirs('models', exist_ok=True)
    with open('models/stratified_ensemble.pkl', 'wb') as f:
        pickle.dump(ensemble_data, f)
    
    print(f"   Saved to models/stratified_ensemble.pkl")
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    
    for name, smape_score in range_smapes.items():
        n_samples = stratified_models[name]['n_samples']
        price_range = stratified_models[name]['price_range']
        print(f"{name:12s}: SMAPE = {smape_score:5.2f}% (n={n_samples:5d}, ${price_range[0]:.0f}-${price_range[1]:.0f})")
    
    weighted_smape = sum(
        range_smapes[name] * stratified_models[name]['n_samples'] 
        for name in range_smapes
    ) / sum(stratified_models[name]['n_samples'] for name in range_smapes)
    
    print(f"\nWeighted Average SMAPE: {weighted_smape:.2f}%")
    print("\n💡 Use this model with predict_stratified.py")
    print("="*70)

if __name__ == "__main__":
    main()
