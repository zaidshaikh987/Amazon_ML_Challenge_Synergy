#!/usr/bin/env python3
"""
Predict using simple robust model with calibration.
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle

sys.path.append('src')
from features import FeatureExtractor, create_feature_matrix
from advanced_features import add_advanced_features_to_df

def main():
    print("="*70)
    print("SIMPLE ROBUST MODEL PREDICTION")
    print("="*70)
    
    # Load model
    print("\n1. Loading model...")
    with open('models/simple_robust.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    calibrator = model_data['calibrator']
    oof_smape = model_data['oof_smape']
    
    print(f"   Model OOF SMAPE: {oof_smape:.2f}%")
    
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
    print("\n6. Making predictions...")
    y_pred_log = model.predict(X)
    
    # Apply calibration
    print("\n7. Applying calibration...")
    y_pred_log_calibrated = calibrator.predict(y_pred_log.reshape(-1, 1))
    
    # Inverse transform
    y_pred = np.expm1(y_pred_log_calibrated)
    
    # Clip to reasonable range (based on training distribution)
    y_pred = np.clip(y_pred, 0.13, 500.0)  # Train range was $0.13-$2796, but clip conservatively
    
    # Fill predictions for all test samples
    y_pred_all = np.full(len(test_df), np.median(y_pred))
    y_pred_all[valid_indices] = y_pred
    
    # Create submission
    submission = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': y_pred_all
    })
    
    # Save
    os.makedirs('submissions', exist_ok=True)
    output_path = 'submissions/simple_robust_predictions.csv'
    submission.to_csv(output_path, index=False)
    
    print(f"\n8. Predictions saved to {output_path}")
    
    # Summary
    print("\n" + "="*70)
    print("PREDICTION SUMMARY")
    print("="*70)
    print(f"Total predictions: {len(submission)}")
    print(f"Price range: ${submission['price'].min():.2f} - ${submission['price'].max():.2f}")
    print(f"Mean price: ${submission['price'].mean():.2f}")
    print(f"Median price: ${submission['price'].median():.2f}")
    print(f"Std: ${submission['price'].std():.2f}")
    
    # Compare to training distribution
    train = pd.read_csv('student_resource/dataset/train.csv')
    print(f"\nTrain mean: ${train['price'].mean():.2f}, std: ${train['price'].std():.2f}")
    print(f"Pred mean:  ${submission['price'].mean():.2f}, std: ${submission['price'].std():.2f}")
    
    if abs(submission['price'].mean() - train['price'].mean()) < 10:
        print("\n✅ Prediction distribution matches training")
    else:
        print("\n⚠️ Prediction distribution differs from training")
    
    print("="*70)

if __name__ == "__main__":
    main()
