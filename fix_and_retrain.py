#!/usr/bin/env python3
"""
Fix the existing approach to get SMAPE below 48%.
No architecture change - just fixing bugs and tuning.
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

# Add src to path
sys.path.append('src')

print("="*70)
print("FIXING EXISTING MODEL TO GET SMAPE < 48%")
print("="*70)

def smape(y_true, y_pred):
    """Calculate SMAPE in original price space."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0
    if not np.any(mask):
        return 0.0
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100

def load_embeddings_safely(emb_dir, sample_ids):
    """Load embeddings and ensure alignment with sample IDs."""
    import pickle
    
    # Load text embeddings
    text_emb = np.load(os.path.join(emb_dir, 'text_embeddings.npy'))
    image_emb = np.load(os.path.join(emb_dir, 'image_embeddings.npy'))
    
    # Load metadata
    with open(os.path.join(emb_dir, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    
    # Create aligned feature matrix
    features = []
    valid_indices = []
    
    for idx, sid in enumerate(sample_ids):
        # Find embedding index
        emb_idx = np.where(metadata['sample_ids'] == sid)[0]
        if len(emb_idx) > 0:
            emb_idx = emb_idx[0]
            # Combine embeddings
            combined = np.concatenate([text_emb[emb_idx], image_emb[emb_idx]])
            features.append(combined)
            valid_indices.append(idx)
    
    return np.array(features), valid_indices

def extract_simple_features(df):
    """Extract simple engineered features."""
    from features import FeatureExtractor
    
    extractor = FeatureExtractor()
    features = []
    
    for _, row in df.iterrows():
        text = row.get('catalog_content', '')
        if pd.isna(text):
            text = ''
        
        # Extract IPQ
        ipq = extractor.extract_ipq(text)
        if ipq is None:
            ipq = 1
        
        # Text length
        text_len = len(str(text))
        word_count = len(str(text).split())
        
        # Brand detection
        has_brand = any(brand in str(text).lower() for brand in ['nike', 'apple', 'samsung', 'sony'])
        
        features.append([
            ipq,
            np.log1p(ipq),
            text_len,
            word_count,
            float(has_brand)
        ])
    
    return np.array(features)

def train_fixed_model(train_csv, emb_dir, test_csv, test_emb_dir):
    """Train model with proper alignment and evaluation."""
    
    print("\n1. Loading data...")
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    print(f"   Train: {len(train_df)} samples")
    print(f"   Test: {len(test_df)} samples")
    
    # Load embeddings with proper alignment
    print("\n2. Loading embeddings with alignment...")
    X_emb_train, valid_train_idx = load_embeddings_safely(emb_dir, train_df['sample_id'].values)
    X_emb_test, valid_test_idx = load_embeddings_safely(test_emb_dir, test_df['sample_id'].values)
    
    print(f"   Train embeddings: {X_emb_train.shape}")
    print(f"   Test embeddings: {X_emb_test.shape}")
    
    # Extract simple features
    print("\n3. Extracting simple features...")
    X_simple_train = extract_simple_features(train_df)
    X_simple_test = extract_simple_features(test_df)
    
    # Combine features for valid samples only
    X_train = np.hstack([
        X_emb_train,
        X_simple_train[valid_train_idx]
    ])
    
    X_test = np.hstack([
        X_emb_test,
        X_simple_test[valid_test_idx]
    ])
    
    # Get aligned targets
    y_train = train_df['price'].values[valid_train_idx]
    
    print(f"\n4. Final feature shapes:")
    print(f"   X_train: {X_train.shape}")
    print(f"   X_test: {X_test.shape}")
    print(f"   y_train: {y_train.shape}")
    
    # Apply log transform
    y_train_log = np.log1p(y_train)
    
    # Train with optimized hyperparameters
    print("\n5. Training with 5-fold CV...")
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,  # Reduced to prevent overfitting
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 20,
        'lambda_l1': 1.0,  # Strong regularization
        'lambda_l2': 1.0,
        'verbose': -1,
        'random_state': 42,
        'n_estimators': 500
    }
    
    # 5-fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train_log[train_idx], y_train_log[val_idx]
        
        # Train
        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val)
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        models.append(model)
        
        # Predict and evaluate IN ORIGINAL SPACE
        y_pred_log = model.predict(X_val, num_iteration=model.best_iteration)
        y_pred = np.expm1(y_pred_log)
        y_true = np.expm1(y_val)
        
        fold_smape = smape(y_true, y_pred)
        cv_scores.append(fold_smape)
        print(f"   Fold {fold+1}: SMAPE = {fold_smape:.2f}%")
    
    print(f"\n   Mean CV SMAPE: {np.mean(cv_scores):.2f}% ± {np.std(cv_scores):.2f}%")
    
    # Train final model on all data
    print("\n6. Training final model on all data...")
    train_data = lgb.Dataset(X_train, label=y_train_log)
    final_model = lgb.train(params, train_data)
    
    # Make test predictions
    print("\n7. Making test predictions...")
    y_test_pred_log = final_model.predict(X_test)
    y_test_pred = np.expm1(y_test_pred_log)
    
    # Clip extreme predictions
    y_test_pred = np.clip(y_test_pred, 0.01, 10000)
    
    # Fill predictions for all test samples
    all_predictions = np.full(len(test_df), np.median(y_test_pred))
    all_predictions[valid_test_idx] = y_test_pred
    
    # Create submission
    submission = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': all_predictions
    })
    
    # Save everything
    print("\n8. Saving model and predictions...")
    
    # Save model
    model_data = {
        'model': final_model,
        'models': models,  # For ensemble
        'cv_scores': cv_scores,
        'feature_dim': X_train.shape[1]
    }
    
    with open('models/fixed_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    # Save submission
    submission.to_csv('submissions/fixed_predictions.csv', index=False)
    
    print("\n" + "="*70)
    print("RESULTS:")
    print(f"CV SMAPE: {np.mean(cv_scores):.2f}%")
    print(f"Test predictions: min=${all_predictions.min():.2f}, max=${all_predictions.max():.2f}")
    print(f"Test predictions: mean=${all_predictions.mean():.2f}, median=${np.median(all_predictions):.2f}")
    
    if np.mean(cv_scores) < 48:
        print("\n✅ SUCCESS! CV SMAPE is below 48%!")
    else:
        print(f"\n⚠️ CV SMAPE is {np.mean(cv_scores):.2f}%")
        print("\nAdditional fixes to try:")
        print("1. Remove outliers from training data (prices > $5000)")
        print("2. Use ensemble of CV models instead of single model")
        print("3. Add more regularization")
    
    print("="*70)
    
    return final_model, cv_scores

if __name__ == "__main__":
    # Run the fix
    train_fixed_model(
        train_csv='student_resource/dataset/train.csv',
        emb_dir='embeddings/train',
        test_csv='student_resource/dataset/test.csv',
        test_emb_dir='embeddings/test'
    )