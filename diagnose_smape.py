#!/usr/bin/env python3
"""
Diagnostic script to understand why SMAPE is high.
Analyzes predictions, identifies problematic samples, and suggests fixes.
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add src to path
sys.path.append('src')
from features import FeatureExtractor, create_feature_matrix

def smape(y_true, y_pred):
    """Calculate SMAPE metric."""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0
    if not np.any(mask):
        return 0.0
    smape_values = np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]
    return np.mean(smape_values) * 100

def diagnose_predictions(train_csv, emb_dir, model_path):
    """Diagnose prediction issues."""
    
    print("="*60)
    print("SMAPE DIAGNOSTIC REPORT")
    print("="*60)
    
    # Load data
    print("\n1. LOADING DATA...")
    train_df = pd.read_csv(train_csv)
    print(f"   Loaded {len(train_df)} samples")
    
    # Load model
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    model = model_data['model']
    
    # Extract features
    print("\n2. EXTRACTING FEATURES...")
    extractor = FeatureExtractor()
    features_df = extractor.extract_all_features(train_df)
    
    # Create feature matrix
    X, feature_columns, valid_indices = create_feature_matrix(features_df, emb_dir, 'optimized')
    print(f"   Created feature matrix: {X.shape}")
    print(f"   Valid samples: {len(valid_indices)}/{len(train_df)}")
    
    # Get aligned targets
    y_all = train_df['price'].values
    y_true = y_all[valid_indices]
    
    # Make predictions
    print("\n3. MAKING PREDICTIONS...")
    y_pred_log = model.predict(X)
    y_pred = np.expm1(y_pred_log)
    
    # Calculate overall SMAPE
    overall_smape = smape(y_true, y_pred)
    print(f"   Overall SMAPE: {overall_smape:.2f}%")
    
    # Analyze predictions
    print("\n4. ANALYZING PREDICTIONS...")
    
    # Basic statistics
    print("\n   Price Statistics:")
    print(f"   True prices:  min=${y_true.min():.2f}, max=${y_true.max():.2f}, mean=${y_true.mean():.2f}, median=${np.median(y_true):.2f}")
    print(f"   Predictions:  min=${y_pred.min():.2f}, max=${y_pred.max():.2f}, mean=${y_pred.mean():.2f}, median=${np.median(y_pred):.2f}")
    
    # Calculate per-sample SMAPE
    per_sample_smape = []
    for yt, yp in zip(y_true, y_pred):
        denom = (abs(yt) + abs(yp)) / 2
        if denom > 0:
            per_sample_smape.append(abs(yt - yp) / denom * 100)
        else:
            per_sample_smape.append(0)
    per_sample_smape = np.array(per_sample_smape)
    
    # Find problematic samples
    print("\n5. IDENTIFYING PROBLEMATIC SAMPLES...")
    high_error_mask = per_sample_smape > 50  # Samples with >50% SMAPE
    print(f"   Samples with SMAPE > 50%: {high_error_mask.sum()} ({high_error_mask.sum()/len(y_true)*100:.1f}%)")
    
    # Analyze by price range
    print("\n6. ANALYZING BY PRICE RANGE...")
    price_ranges = [(0, 10), (10, 50), (50, 100), (100, 500), (500, float('inf'))]
    
    for low, high in price_ranges:
        mask = (y_true >= low) & (y_true < high)
        if mask.sum() > 0:
            range_smape = smape(y_true[mask], y_pred[mask])
            range_mae = np.mean(np.abs(y_true[mask] - y_pred[mask]))
            print(f"   ${low:3d}-${high:3.0f}: n={mask.sum():5d}, SMAPE={range_smape:6.2f}%, MAE=${range_mae:.2f}")
    
    # Analyze prediction bias
    print("\n7. ANALYZING PREDICTION BIAS...")
    residuals = y_pred - y_true
    print(f"   Mean residual: ${residuals.mean():.2f}")
    print(f"   Median residual: ${np.median(residuals):.2f}")
    print(f"   Std residual: ${residuals.std():.2f}")
    
    if residuals.mean() > 5:
        print("   ⚠️ Model tends to OVERPREDICT prices")
    elif residuals.mean() < -5:
        print("   ⚠️ Model tends to UNDERPREDICT prices")
    else:
        print("   ✓ Model has minimal systematic bias")
    
    # Check for outliers
    print("\n8. CHECKING FOR OUTLIERS...")
    pred_outliers = (y_pred > np.percentile(y_true, 99)) | (y_pred < np.percentile(y_true, 1))
    print(f"   Prediction outliers: {pred_outliers.sum()} samples")
    
    # Feature importance analysis
    print("\n9. TOP FEATURES...")
    if 'feature_importance' in model_data:
        importance = model_data['feature_importance']
        top_features = sorted(zip(feature_columns, importance), key=lambda x: x[1], reverse=True)[:10]
        for i, (feat, imp) in enumerate(top_features, 1):
            print(f"   {i:2d}. {feat[:40]:40s}: {imp:.1f}")
    
    # Specific issues and recommendations
    print("\n10. DIAGNOSIS & RECOMMENDATIONS:")
    print("="*60)
    
    issues = []
    
    # Issue 1: High SMAPE
    if overall_smape > 30:
        issues.append("CRITICAL: Very high SMAPE (>30%)")
        print("\n❌ CRITICAL: Very high SMAPE")
        print("   Likely causes:")
        print("   - Model trained on log-space but evaluated on original space")
        print("   - Missing or poor quality embeddings")
        print("   - Insufficient regularization")
    elif overall_smape > 20:
        issues.append("WARNING: High SMAPE (20-30%)")
        print("\n⚠️ WARNING: High SMAPE")
        print("   Potential improvements:")
        print("   - Tune hyperparameters")
        print("   - Add more features")
        print("   - Try ensemble methods")
    else:
        print("\n✅ GOOD: SMAPE is below 20%")
    
    # Issue 2: Missing embeddings
    missing_ratio = 1 - len(valid_indices)/len(train_df)
    if missing_ratio > 0.1:
        issues.append(f"Missing embeddings for {missing_ratio*100:.1f}% samples")
        print(f"\n❌ CRITICAL: Missing embeddings for {missing_ratio*100:.1f}% of samples")
        print("   Fix: Re-run embedding extraction for all samples")
    
    # Issue 3: Price distribution mismatch
    if abs(y_pred.mean() - y_true.mean()) > 50:
        issues.append("Large difference in mean prices")
        print("\n❌ Price distribution mismatch")
        print("   The predicted price distribution doesn't match the true distribution")
        print("   Fix: Check target transformation and inverse transformation")
    
    # Issue 4: Outlier predictions
    if pred_outliers.sum() > len(y_true) * 0.05:
        issues.append(f"Too many outlier predictions ({pred_outliers.sum()})")
        print(f"\n⚠️ Too many outlier predictions: {pred_outliers.sum()}")
        print("   Fix: Clip predictions to reasonable range or add outlier detection")
    
    print("\n" + "="*60)
    print("SUMMARY OF ISSUES:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    
    if not issues:
        print("  No major issues found!")
    
    print("\nACTION PLAN TO IMPROVE SMAPE:")
    print("1. Ensure embeddings exist for ALL samples")
    print("2. Verify metrics are calculated in original price space")
    print("3. Try improved hyperparameters with stronger regularization")
    print("4. Consider ensemble of multiple models")
    print("5. Add post-processing: clip extreme predictions")
    print("="*60)
    
    return {
        'overall_smape': overall_smape,
        'y_true': y_true,
        'y_pred': y_pred,
        'issues': issues
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Diagnose SMAPE issues')
    parser.add_argument('--train_csv', default='student_resource/dataset/train.csv', help='Training CSV')
    parser.add_argument('--emb_dir', default='embeddings/train', help='Embeddings directory')
    parser.add_argument('--model', default='models/baseline_lgb.pkl', help='Model path')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.train_csv):
        print(f"Error: Training CSV not found: {args.train_csv}")
        sys.exit(1)
    
    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        print("Please train a model first using train_baseline.py or train_improved_v2.py")
        sys.exit(1)
    
    # Run diagnosis
    results = diagnose_predictions(args.train_csv, args.emb_dir, args.model)
    
    print(f"\nFinal SMAPE: {results['overall_smape']:.2f}%")