#!/usr/bin/env python3
"""
Load trained model and generate predictions for test data.
Produces test_out.csv in the required format.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(model_path):
    """
    Load trained model and metadata.
    
    Args:
        model_path: Path to saved model
    
    Returns:
        dict: Model data
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Import models if needed
    try:
        from sentence_transformers import SentenceTransformer
        from transformers import CLIPModel, CLIPProcessor
    except ImportError:
        pass
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    logger.info(f"Loaded model from {model_path}")
    logger.info(f"Model type: {type(model_data['model'])}")
    logger.info(f"Target transform: {model_data['target_transform']}")
    logger.info(f"Feature count: {len(model_data['feature_columns'])}")
    
    return model_data

def inverse_transform_target(y_pred, transform_type):
    """
    Apply inverse transformation to predictions.
    
    Args:
        y_pred: Predictions
        transform_type: Type of transformation used
    
    Returns:
        numpy array: Inverse transformed predictions
    """
    if transform_type == 'log1p':
        return np.expm1(y_pred)
    elif transform_type == 'log':
        return np.exp(y_pred)
    elif transform_type == 'sqrt':
        return np.square(y_pred)
    else:
        logger.warning(f"Unknown transform type: {transform_type}, returning predictions as-is")
        return y_pred

def ensure_positive_prices(prices, min_price=0.01):
    """
    Ensure all predicted prices are positive.
    
    Args:
        prices: Predicted prices
        min_price: Minimum allowed price
    
    Returns:
        numpy array: Adjusted prices
    """
    prices = np.array(prices)
    
    # Set negative prices to minimum
    prices[prices < min_price] = min_price
    
    # Set zero prices to minimum
    prices[prices == 0] = min_price
    
    return prices

def main():
    parser = argparse.ArgumentParser(description='Generate predictions using trained model')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--test_csv', required=True, help='Test CSV file')
    parser.add_argument('--emb_dir', required=True, help='Embeddings directory')
    parser.add_argument('--out', required=True, help='Output CSV file path')
    parser.add_argument('--id_col', default='sample_id', help='ID column name')
    parser.add_argument('--max_price', type=float, default=3000.0, 
                         help='Maximum allowed price for clipping')
    parser.add_argument('--min_price', type=float, default=0.01, help='Minimum allowed price')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.test_csv):
        logger.error(f"Test CSV not found: {args.test_csv}")
        sys.exit(1)
    
    if not os.path.exists(args.emb_dir):
        logger.error(f"Embeddings directory not found: {args.emb_dir}")
        sys.exit(1)
    
    try:
        # Load model
        logger.info("Loading trained model")
        model_data = load_model(args.model)
        
        # Load model
        model = model_data['model']
        feature_columns = model_data['feature_columns']
        target_transform = model_data['target_transform']
        model_name = model_data['model_name']
        
        # Load test data
        logger.info(f"Loading test data from {args.test_csv}")
        test_df = pd.read_csv(args.test_csv)
        logger.info(f"Loaded {len(test_df)} test samples")
        
        # Check required columns
        required_cols = [args.id_col, 'catalog_content']
        missing_cols = [col for col in required_cols if col not in test_df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            sys.exit(1)
        
        logger.info("Extracting features")
        from features import FeatureExtractor, create_feature_matrix
        
        extractor = FeatureExtractor()
        # Pass image directory to check image availability
        image_dir = os.path.join(os.path.dirname(args.emb_dir), '..', 'images', os.path.basename(args.emb_dir))
        if not os.path.exists(image_dir):
            image_dir = None
        features_df = extractor.extract_all_features(test_df, image_dir=image_dir)
        # Create feature matrix
        logger.info("Creating feature matrix")
        # Use 'simple' prefix for embeddings to match training
        emb_prefix = 'simple_' if model_name == 'simple' else ''
        X, _, valid_indices = create_feature_matrix(features_df, args.emb_dir, model_name)
        
        # Generate predictions
        logger.info("Generating predictions")
        y_pred_log = model.predict(X)
        
        y_pred = inverse_transform_target(y_pred_log, target_transform)

        # Clip predictions to reasonable range
        logger.info(f"Clipping predictions to [{args.min_price}, {args.max_price}]")
        y_pred = np.clip(y_pred, args.min_price, args.max_price)

        # Ensure no negative prices (redundant with clip but kept for safety)
        y_pred = np.maximum(y_pred, args.min_price)
        
        # Ensure positive prices
        y_pred = ensure_positive_prices(y_pred, args.min_price)
        
        # Create output DataFrame
        output_df = pd.DataFrame({
            args.id_col: test_df[args.id_col],
            'price': y_pred
        })
        
        # Validate output
        logger.info(f"Generated {len(output_df)} predictions")
        logger.info(f"Price range: {y_pred.min():.4f} - {y_pred.max():.4f}")
        logger.info(f"Mean price: {y_pred.mean():.4f}")
        logger.info(f"Median price: {np.median(y_pred):.4f}")
        
        # Check for any remaining issues
        if np.any(y_pred <= 0) or np.any(y_pred > args.max_price * 1.1):
           logger.warning("Some predictions are out of expected range!")
           logger.warning(f"Price range: {y_pred.min()} - {y_pred.max()}")
        
        # Save predictions
        output_df.to_csv(args.out, index=False)
        logger.info(f"Predictions saved to {args.out}")
        
        # Print sample predictions
        logger.info("Sample predictions:")
        for i, (_, row) in enumerate(output_df.head(10).iterrows()):
            logger.info(f"  {row[args.id_col]}: ${row['price']:.2f}")
        
        # Validate output format
        expected_cols = [args.id_col, 'price']
        if list(output_df.columns) != expected_cols:
            logger.error(f"Output columns don't match expected format: {expected_cols}")
            sys.exit(1)
        
        if len(output_df) != len(test_df):
            logger.error(f"Output length ({len(output_df)}) doesn't match test length ({len(test_df)})")
            sys.exit(1)
        
        logger.info("Prediction completed successfully!")
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
