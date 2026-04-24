#!/usr/bin/env python3
"""
Simple robust model: Advanced features + Single LightGBM + Heavy regularization.
NO stratification (causes overfitting).
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

def main():
    print("="*70)
    print("SIMPLE ROBUST MODEL WITH ADVANCED FEATURES")
    print("="*70)
    
    # Load data
    print("\n1. Loading training data...")
    train_df = pd.read_csv('student_resource/dataset/train.csv')
    print(f"   Loaded {len(train_df)} samples")
    
    # Extract features
    print("\n2. Extracting basic features...")
    extractor = FeatureExtractor()
    train_df = extractor.extract_all_features(train_df)
    
    print("\n3. Extracting advanced features...")
    train_df = add_advanced_features_to_df(train_df)
    
    # Create feature matrix
    print("\n4. Creating feature matrix...")
    X, feature_columns, valid_indices = create_feature_matrix(train_df, 'embeddings/train', 'optimized')
    
    # Get prices
    y_all = train_df['price'].values
    y = y_all[valid_indices]
    y_log = np.log1p(y)
    
    print(f"   Feature matrix: {X.shape}")
    print(f"   Valid samples: {len(y)}")
    print(f"   Price range: ${y.min():.2f} - ${y.max():.2f}")
    
    # ROBUST hyperparameters - HEAVY regularization to prevent overfitting
    print("\n5. Training single robust model with HEAVY regularization...")
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,  # Small - prevent overfitting
        'max_depth': 6,    # Shallow trees
        'learning_rate': 0.01,  # Very slow learning
        'feature_fraction': 0.6,  # Aggressive feature sampling
        'bagging_fraction': 0.7,  # Aggressive row sampling
        'bagging_freq': 5,
        'min_data_in_leaf': 100,  # High - require many samples per leaf
        'min_sum_hessian_in_leaf': 20.0,  # High - strong regularization
        'lambda_l1': 2.0,  # Very strong L1
        'lambda_l2': 2.0,  # Very strong L2
        'min_gain_to_split': 0.1,  # Require significant gain to split
        'verbose': -1,
        'random_state': 42,
        'n_estimators': 3000  # More iterations but will early stop
    }
    
    # 5-fold CV with proper OOF predictions
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    oof_preds_log = np.zeros(len(y))
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n  Fold {fold+1}/5...")
        
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
        cv_scores.append(fold_smape)
        print(f"    SMAPE: {fold_smape:.2f}%, Iterations: {model.best_iteration}")
    
    # Calculate OOF SMAPE (most reliable estimate)
    oof_preds = np.expm1(oof_preds_log)
    oof_smape = smape(y, oof_preds)
    
    print(f"\n  Fold SMAPE: {np.mean(cv_scores):.2f}% ± {np.std(cv_scores):.2f}%")
    print(f"  OOF SMAPE:  {oof_smape:.2f}% (most reliable)")
    
    # Train final model on all data
    print("\n6. Training final model on full data...")
    train_data = lgb.Dataset(X, label=y_log)
    final_model = lgb.train(params, train_data)
    
    # Calibration: fit simple correction on OOF predictions
    print("\n7. Calibrating predictions...")
    from sklearn.linear_model import Ridge
    
    # Fit calibrator on OOF predictions
    calibrator = Ridge(alpha=10.0)
    calibrator.fit(oof_preds_log.reshape(-1, 1), y_log)
    
    print(f"   Calibration coef: {calibrator.coef_[0]:.4f}")
    print(f"   Calibration intercept: {calibrator.intercept_:.4f}")
    
    # Save everything
    print("\n8. Saving model...")
    
    model_data = {
        'model': final_model,
        'calibrator': calibrator,
        'feature_columns': feature_columns,
        'target_transform': 'log1p',
        'model_name': 'robust',
        'oof_smape': oof_smape,
        'cv_smapes': cv_scores,
        'oof_preds': oof_preds
    }
    
    os.makedirs('models', exist_ok=True)
    with open('models/simple_robust.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"   Saved to models/simple_robust.pkl")
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"OOF SMAPE: {oof_smape:.2f}%")
    print(f"CV SMAPE:  {np.mean(cv_scores):.2f}% ± {np.std(cv_scores):.2f}%")
    print(f"\n✅ Model trained with heavy regularization")
    print(f"✅ Calibration fitted on OOF predictions")
    print(f"✅ Ready for test predictions")
    print("\nRun: python predict_simple_robust.py")
    print("="*70)

if __name__ == "__main__":
    main()
