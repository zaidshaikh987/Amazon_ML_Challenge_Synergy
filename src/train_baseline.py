#!/usr/bin/env python3
"""
Train baseline LightGBM model with log-target transformation and 5-fold CV.
Uses text embeddings, image embeddings, and engineered features.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def smape(y_true, y_pred):
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE).
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        float: SMAPE score
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Avoid division by zero
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0
    
    if not np.any(mask):
        return 0.0
    
    smape_values = np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]
    return np.mean(smape_values) * 100

def create_price_bins(y, n_bins=5):
    """
    Create price bins for stratified CV.
    
    Args:
        y: Target values
        n_bins: Number of bins
    
    Returns:
        numpy array: Bin labels
    """
    return pd.qcut(y, q=n_bins, labels=False, duplicates='drop')

def train_lightgbm_model(X, y, cv_folds=5, quick=False):
    """
    Train LightGBM model with cross-validation.
    
    Args:
        X: Feature matrix
        y: Target values
        cv_folds: Number of CV folds
        quick: Whether to use quick training (fewer iterations)
    
    Returns:
        tuple: (trained_model, cv_scores, feature_importance)
    """
    logger.info(f"Training LightGBM model with {cv_folds}-fold CV")
    logger.info(f"Feature matrix shape: {X.shape}")
    
    # LightGBM parameters (OPTIMIZED for low SMAPE!)
    if quick:
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 127,  # Increased complexity
            'max_depth': 12,
            'learning_rate': 0.03,  # Lower for better convergence
            'feature_fraction': 0.85,
            'bagging_fraction': 0.85,
            'bagging_freq': 5,
            'min_data_in_leaf': 15,
            'min_child_weight': 0.001,
            'lambda_l1': 0.5,  # Stronger L1 regularization
            'lambda_l2': 0.5,  # Stronger L2 regularization
            'min_gain_to_split': 0.01,
            'verbose': -1,
            'random_state': 42,
            'n_estimators': 300
        }
    else:
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 255,  # Much higher for complex patterns
            'max_depth': 15,
            'learning_rate': 0.01,  # Very low for fine-grained learning
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_data_in_leaf': 10,
            'min_child_weight': 0.001,
            'lambda_l1': 1.0,  # Strong regularization
            'lambda_l2': 1.0,
            'min_gain_to_split': 0.01,
            'max_bin': 255,
            'verbose': -1,
            'random_state': 42,
            'n_estimators': 2000  # More iterations for better fit
        }
    
    # Create price bins for stratified CV
    price_bins = create_price_bins(y, n_bins=cv_folds)
    
    # Cross-validation
    cv_scores = {
        'smape': [],
        'mae': [],
        'rmse': [],
        'r2': []
    }
    
    if cv_folds > 1:
        # Use stratified CV if possible, otherwise regular CV
        try:
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_splits = skf.split(X, price_bins)
        except:
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_splits = kf.split(X)
        
        fold_models = []
        
        for fold, (train_idx, val_idx) in enumerate(cv_splits):
            logger.info(f"Training fold {fold + 1}/{cv_folds}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create LightGBM datasets
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # Train model
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )
            
            # Predictions
            y_pred = model.predict(X_val, num_iteration=model.best_iteration)
            
            # Calculate metrics
            fold_smape = smape(y_val, y_pred)
            fold_mae = mean_absolute_error(y_val, y_pred)
            fold_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            fold_r2 = r2_score(y_val, y_pred)
            
            cv_scores['smape'].append(fold_smape)
            cv_scores['mae'].append(fold_mae)
            cv_scores['rmse'].append(fold_rmse)
            cv_scores['r2'].append(fold_r2)
            
            fold_models.append(model)
            
            logger.info(f"Fold {fold + 1} - SMAPE: {fold_smape:.4f}, MAE: {fold_mae:.4f}, RMSE: {fold_rmse:.4f}, R²: {fold_r2:.4f}")
        
        # Train final model on full dataset
        logger.info("Training final model on full dataset")
        train_data = lgb.Dataset(X, label=y)
        final_model = lgb.train(params, train_data)
        
        # Calculate average CV scores
        avg_scores = {metric: np.mean(scores) for metric, scores in cv_scores.items()}
        std_scores = {metric: np.std(scores) for metric, scores in cv_scores.items()}
        
        logger.info("Cross-validation results:")
        for metric in ['smape', 'mae', 'rmse', 'r2']:
            logger.info(f"  {metric.upper()}: {avg_scores[metric]:.4f} ± {std_scores[metric]:.4f}")
        
        # Feature importance from final model
        feature_importance = final_model.feature_importance(importance_type='gain')
        
    else:
        # No CV, train on full dataset
        logger.info("Training model on full dataset (no CV)")
        train_data = lgb.Dataset(X, label=y)
        final_model = lgb.train(params, train_data)
        feature_importance = final_model.feature_importance(importance_type='gain')
        
        # Calculate metrics on full dataset
        y_pred = final_model.predict(X)
        avg_scores = {
            'smape': smape(y, y_pred),
            'mae': mean_absolute_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'r2': r2_score(y, y_pred)
        }
        
        logger.info("Training results:")
        for metric, score in avg_scores.items():
            logger.info(f"  {metric.upper()}: {score:.4f}")
    
    return final_model, avg_scores, feature_importance

def main():
    parser = argparse.ArgumentParser(description='Train baseline LightGBM model')
    parser.add_argument('--train_csv', required=True, help='Training CSV file')
    parser.add_argument('--emb_dir', required=True, help='Embeddings directory')
    parser.add_argument('--out_dir', required=True, help='Output directory for model')
    parser.add_argument('--model', default='optimized', choices=['optimized'], help='Model name')
    parser.add_argument('--cv_folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--quick', type=int, default=0, help='Quick training mode')
    parser.add_argument('--target_col', default='price', help='Target column name')
    parser.add_argument('--id_col', default='sample_id', help='ID column name')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.train_csv):
        logger.error(f"Training CSV not found: {args.train_csv}")
        sys.exit(1)
    
    if not os.path.exists(args.emb_dir):
        logger.error(f"Embeddings directory not found: {args.emb_dir}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    try:
        # Load training data
        logger.info(f"Loading training data from {args.train_csv}")
        train_df = pd.read_csv(args.train_csv)
        logger.info(f"Loaded {len(train_df)} training samples")
        
        # Check required columns
        required_cols = [args.id_col, args.target_col, 'catalog_content']
        missing_cols = [col for col in required_cols if col not in train_df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            sys.exit(1)
        
        # Extract features
        logger.info("Extracting features")
        from features import FeatureExtractor, create_feature_matrix
        
        extractor = FeatureExtractor()
        # Pass image directory to check image availability
        image_dir = os.path.join(os.path.dirname(args.emb_dir), '..', 'images', os.path.basename(args.emb_dir))
        if not os.path.exists(image_dir):
            image_dir = None
        features_df = extractor.extract_all_features(train_df, image_dir=image_dir)
        
        # Create feature matrix
        X, feature_columns, valid_indices = create_feature_matrix(features_df, args.emb_dir, args.model)
        
        # Filter target to match valid samples
        y_all = train_df[args.target_col].values
        y = y_all[valid_indices]
        
        # Log-transform target for better performance
        y_log = np.log1p(y)
        logger.info(f"Target range: {y.min():.2f} - {y.max():.2f}")
        logger.info(f"Log-target range: {y_log.min():.2f} - {y_log.max():.2f}")
        
        # Train model
        quick_mode = args.quick == 1
        model, scores, feature_importance = train_lightgbm_model(
            X, y_log, cv_folds=args.cv_folds, quick=quick_mode
        )
        
        # Save model
        model_path = os.path.join(args.out_dir, 'baseline_lgb.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'feature_columns': feature_columns,
                'target_transform': 'log1p',
                'model_name': args.model,
                'cv_scores': scores,
                'feature_importance': feature_importance
            }, f)
        
        logger.info(f"Model saved to {model_path}")
        
        # Save feature importance
        importance_df = pd.DataFrame({
            'feature': feature_columns,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        importance_path = os.path.join(args.out_dir, 'feature_importance.csv')
        importance_df.to_csv(importance_path, index=False)
        logger.info(f"Feature importance saved to {importance_path}")
        
        # Print top features
        logger.info("Top 20 most important features:")
        for i, (_, row) in enumerate(importance_df.head(20).iterrows()):
            logger.info(f"  {i+1:2d}. {row['feature']}: {row['importance']:.4f}")
        
        # Print final results
        logger.info("Training completed successfully!")
        logger.info(f"CV SMAPE: {scores['smape']:.4f}%")
        logger.info(f"CV MAE: {scores['mae']:.4f}")
        logger.info(f"CV RMSE: {scores['rmse']:.4f}")
        logger.info(f"CV R²: {scores['r2']:.4f}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
