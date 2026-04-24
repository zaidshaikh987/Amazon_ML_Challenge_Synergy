#!/usr/bin/env python3
"""
Improved training pipeline with better feature engineering and hyperparameters.
Goal: Reduce SMAPE below 20%
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import pickle
import logging
import warnings
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb

# Add src to path
sys.path.append('src')
from features import FeatureExtractor, create_feature_matrix

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def smape(y_true, y_pred):
    """Calculate SMAPE metric."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0
    
    if not np.any(mask):
        return 0.0
    
    smape_values = np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]
    return np.mean(smape_values) * 100

def create_price_bins(y, n_bins=5):
    """Create price bins for stratified CV."""
    return pd.qcut(y, q=n_bins, labels=False, duplicates='drop')

def add_advanced_features(df):
    """Add more sophisticated features."""
    df = df.copy()
    
    # Log transform of numerical features
    if 'ipq' in df.columns:
        df['ipq_log'] = np.log1p(df['ipq'])
        df['ipq_sqrt'] = np.sqrt(df['ipq'])
    
    # Text complexity features
    if 'text_length' in df.columns:
        df['text_log'] = np.log1p(df['text_length'])
        df['words_per_char'] = df['word_count'] / (df['char_count'] + 1)
    
    # Interaction features
    if 'has_premium_brand' in df.columns and 'ipq' in df.columns:
        df['premium_ipq_interaction'] = df['has_premium_brand'] * df['ipq']
    
    # Category indicators as numerical
    category_cols = [col for col in df.columns if col.startswith('category_')]
    for col in category_cols:
        df[f'{col}_numeric'] = df[col].astype(int)
    
    return df

def train_improved_lightgbm(X, y, cv_folds=5):
    """Train LightGBM with improved hyperparameters."""
    
    logger.info(f"Training improved LightGBM with {cv_folds}-fold CV")
    logger.info(f"Feature matrix shape: {X.shape}")
    
    # Improved hyperparameters based on typical e-commerce data patterns
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 127,  # Increased for more complex patterns
        'learning_rate': 0.03,  # Slightly lower for better generalization
        'feature_fraction': 0.7,  # More aggressive feature sampling
        'bagging_fraction': 0.7,  # More aggressive row sampling
        'bagging_freq': 5,
        'min_data_in_leaf': 30,  # Higher to prevent overfitting
        'min_sum_hessian_in_leaf': 5.0,
        'lambda_l1': 0.5,  # Stronger regularization
        'lambda_l2': 0.5,
        'max_depth': -1,  # No limit, controlled by num_leaves
        'verbose': -1,
        'random_state': 42,
        'n_estimators': 2000,  # More iterations with early stopping
        'early_stopping_rounds': 100
    }
    
    # Price bins for stratified CV
    price_bins = create_price_bins(np.expm1(y), n_bins=min(cv_folds, 5))
    
    # Cross-validation
    cv_scores = {'smape': [], 'mae': [], 'rmse': [], 'r2': []}
    models = []
    
    # Use stratified KFold based on price bins
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, price_bins)):
        logger.info(f"Training fold {fold + 1}/{cv_folds}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train model with early stopping
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=params['early_stopping_rounds'], verbose=False),
                lgb.log_evaluation(period=0)
            ]
        )
        
        models.append(model)
        
        # Predictions in log space
        y_pred_log = model.predict(X_val, num_iteration=model.best_iteration)
        
        # Convert to original space for metrics
        y_val_orig = np.expm1(y_val)
        y_pred_orig = np.expm1(y_pred_log)
        
        # Clip extreme predictions
        y_pred_orig = np.clip(y_pred_orig, 0.01, np.percentile(y_val_orig, 99.5))
        
        # Calculate metrics
        fold_smape = smape(y_val_orig, y_pred_orig)
        fold_mae = mean_absolute_error(y_val_orig, y_pred_orig)
        fold_rmse = np.sqrt(mean_squared_error(y_val_orig, y_pred_orig))
        fold_r2 = r2_score(y_val_orig, y_pred_orig)
        
        cv_scores['smape'].append(fold_smape)
        cv_scores['mae'].append(fold_mae)
        cv_scores['rmse'].append(fold_rmse)
        cv_scores['r2'].append(fold_r2)
        
        logger.info(f"Fold {fold + 1} - SMAPE: {fold_smape:.2f}%, MAE: {fold_mae:.2f}, RMSE: {fold_rmse:.2f}, R²: {fold_r2:.4f}")
    
    # Calculate average scores
    avg_scores = {metric: np.mean(scores) for metric, scores in cv_scores.items()}
    std_scores = {metric: np.std(scores) for metric, scores in cv_scores.items()}
    
    logger.info("\n" + "="*50)
    logger.info("Cross-validation results (original price space):")
    for metric in ['smape', 'mae', 'rmse', 'r2']:
        logger.info(f"  {metric.upper()}: {avg_scores[metric]:.2f} ± {std_scores[metric]:.2f}")
    logger.info("="*50 + "\n")
    
    # Train final model on full dataset with best parameters
    logger.info("Training final model on full dataset...")
    
    # Use average of best iterations from CV
    best_iterations = [m.best_iteration for m in models]
    params['n_estimators'] = int(np.mean(best_iterations) * 1.2)  # 20% more iterations for full data
    
    train_data = lgb.Dataset(X, label=y)
    final_model = lgb.train(params, train_data)
    
    # Get feature importance
    feature_importance = final_model.feature_importance(importance_type='gain')
    
    return final_model, avg_scores, feature_importance, models

def ensemble_predict(models, X):
    """Make ensemble predictions from multiple models."""
    predictions = []
    for model in models:
        pred = model.predict(X, num_iteration=model.best_iteration)
        predictions.append(pred)
    
    # Use median for robustness
    return np.median(predictions, axis=0)

def main():
    parser = argparse.ArgumentParser(description='Train improved model')
    parser.add_argument('--train_csv', required=True, help='Training CSV file')
    parser.add_argument('--emb_dir', required=True, help='Embeddings directory')
    parser.add_argument('--out_dir', required=True, help='Output directory')
    parser.add_argument('--cv_folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--ensemble', action='store_true', help='Save ensemble model')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load training data
    logger.info(f"Loading training data from {args.train_csv}")
    train_df = pd.read_csv(args.train_csv)
    logger.info(f"Loaded {len(train_df)} training samples")
    
    # Extract features
    logger.info("Extracting features...")
    extractor = FeatureExtractor()
    features_df = extractor.extract_all_features(train_df)
    
    # Add advanced features
    logger.info("Adding advanced features...")
    features_df = add_advanced_features(features_df)
    
    # Create feature matrix
    logger.info("Creating feature matrix...")
    X, feature_columns, valid_indices = create_feature_matrix(features_df, args.emb_dir, 'optimized')
    
    # Get aligned targets
    y_all = train_df['price'].values
    y = y_all[valid_indices]
    
    # Log transform target
    y_log = np.log1p(y)
    
    logger.info(f"Training on {len(y)} samples with {X.shape[1]} features")
    logger.info(f"Target range: ${y.min():.2f} - ${y.max():.2f}")
    logger.info(f"Log target range: {y_log.min():.2f} - {y_log.max():.2f}")
    
    # Train model
    final_model, scores, feature_importance, cv_models = train_improved_lightgbm(
        X, y_log, cv_folds=args.cv_folds
    )
    
    # Save model
    model_path = os.path.join(args.out_dir, 'improved_model.pkl')
    
    model_data = {
        'model': final_model,
        'feature_columns': feature_columns,
        'target_transform': 'log1p',
        'model_name': 'improved',
        'cv_scores': scores,
        'feature_importance': feature_importance
    }
    
    if args.ensemble:
        model_data['cv_models'] = cv_models
        logger.info(f"Saving ensemble of {len(cv_models)} models")
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    logger.info(f"Model saved to {model_path}")
    
    # Save feature importance
    importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    importance_path = os.path.join(args.out_dir, 'improved_feature_importance.csv')
    importance_df.to_csv(importance_path, index=False)
    
    # Print top features
    logger.info("\nTop 20 most important features:")
    for i, row in importance_df.head(20).iterrows():
        logger.info(f"  {i+1:2d}. {row['feature']}: {row['importance']:.1f}")
    
    logger.info(f"\nFinal CV SMAPE: {scores['smape']:.2f}%")
    
    if scores['smape'] < 20:
        logger.info("✅ SUCCESS! SMAPE is below 20%!")
    else:
        logger.info(f"⚠️ SMAPE is {scores['smape']:.2f}%, still need improvement")
        logger.info("\nSuggestions to improve:")
        logger.info("1. Check if embeddings are properly generated for all samples")
        logger.info("2. Try ensemble with XGBoost or CatBoost")
        logger.info("3. Add more domain-specific features")
        logger.info("4. Consider outlier removal in training data")

if __name__ == "__main__":
    main()