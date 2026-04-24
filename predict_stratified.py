#!/usr/bin/env python3
"""
Predict using stratified ensemble.
Routes predictions to appropriate price-range-specific models.
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle

sys.path.append('src')
from features import FeatureExtractor, create_feature_matrix
from advanced_features import add_advanced_features_to_df

def predict_with_stratified_ensemble(ensemble_data, X):
    """Make predictions using stratified ensemble."""
    
    # First, use global model to estimate which price range each sample belongs to
    global_model = ensemble_data['global_model']
    initial_preds_log = global_model.predict(X)
    initial_preds = np.expm1(initial_preds_log)
    
    # Initialize final predictions
    final_preds = np.zeros(len(X))
    
    # Route each sample to appropriate model
    for name, model_data in ensemble_data['stratified_models'].items():
        model = model_data['model']
        low, high = model_data['price_range']
        
        # Find samples that belong to this range (based on initial estimate)
        mask = (initial_preds >= low) & (initial_preds < high)
        
        if mask.sum() > 0:
            # Use range-specific model
            y_pred_log = model.predict(X[mask])
            y_pred = np.expm1(y_pred_log)
            final_preds[mask] = y_pred
    
    # For any samples not assigned, use global model
    unassigned = final_preds == 0
    if unassigned.sum() > 0:
        final_preds[unassigned] = initial_preds[unassigned]
    
    return final_preds

def main():
    print("="*70)
    print("STRATIFIED ENSEMBLE PREDICTION")
    print("="*70)
    
    # Load ensemble
    print("\n1. Loading stratified ensemble...")
    with open('models/stratified_ensemble.pkl', 'rb') as f:
        ensemble_data = pickle.load(f)
    
    print(f"   Loaded {len(ensemble_data['stratified_models'])} stratified models")
    
    # Load test data
    print("\n2. Loading test data...")
    test_df = pd.read_csv('student_resource/dataset/test.csv')
    print(f"   Loaded {len(test_df)} test samples")
    
    # Extract features (same as training)
    print("\n3. Extracting features...")
    extractor = FeatureExtractor()
    test_df = extractor.extract_all_features(test_df)
    
    print("\n4. Extracting advanced features...")
    test_df = add_advanced_features_to_df(test_df)
    
    print("\n5. Creating feature matrix...")
    X, _, valid_indices = create_feature_matrix(test_df, 'embeddings/test', 'optimized')
    
    print(f"   Feature matrix: {X.shape}")
    print(f"   Valid samples: {len(valid_indices)}")
    
    # Make predictions
    print("\n6. Making stratified predictions...")
    y_pred_valid = predict_with_stratified_ensemble(ensemble_data, X)
    
    # Clip to reasonable range
    y_pred_valid = np.clip(y_pred_valid, 0.01, 5000.0)
    
    # Fill predictions for all test samples
    y_pred_all = np.full(len(test_df), np.median(y_pred_valid))
    y_pred_all[valid_indices] = y_pred_valid
    
    # Create submission
    submission = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': y_pred_all
    })
    
    # Save
    os.makedirs('submissions', exist_ok=True)
    output_path = 'submissions/stratified_ensemble_predictions.csv'
    submission.to_csv(output_path, index=False)
    
    print(f"\n7. Predictions saved to {output_path}")
    
    print("\n" + "="*70)
    print("PREDICTION SUMMARY")
    print("="*70)
    print(f"Total predictions: {len(submission)}")
    print(f"Price range: ${submission['price'].min():.2f} - ${submission['price'].max():.2f}")
    print(f"Mean price: ${submission['price'].mean():.2f}")
    print(f"Median price: ${submission['price'].median():.2f}")
    print("="*70)

if __name__ == "__main__":
    main()
