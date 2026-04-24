#!/usr/bin/env python3
"""
Blend k-NN predictions with LightGBM predictions.
"""
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def blend_predictions(lgb_file, knn_file, output_file, lgb_weight=0.6, knn_weight=0.4):
    """
    Blend two prediction files with specified weights.
    
    Args:
        lgb_file: Path to LightGBM predictions
        knn_file: Path to k-NN predictions
        output_file: Path to save blended predictions
        lgb_weight: Weight for LightGBM predictions
        knn_weight: Weight for k-NN predictions
    """
    logger.info(f"Loading LightGBM predictions from {lgb_file}")
    lgb = pd.read_csv(lgb_file)
    logger.info(f"LightGBM predictions shape: {lgb.shape}")
    
    logger.info(f"Loading k-NN predictions from {knn_file}")
    knn = pd.read_csv(knn_file)
    logger.info(f"k-NN predictions shape: {knn.shape}")
    
    # Load test data for sample_ids
    test = pd.read_csv("student_resource/dataset/test.csv")
    
    # Merge predictions
    logger.info("Merging predictions...")
    merged = test[['sample_id']].copy()
    merged = merged.merge(lgb, on='sample_id', how='left')
    merged = merged.merge(knn, on='sample_id', how='left', suffixes=('_lgb', '_knn'))
    
    # Check column names
    logger.info(f"Merged columns: {merged.columns.tolist()}")
    
    # Blend predictions
    if 'price' in merged.columns and 'knn_price' in merged.columns:
        logger.info(f"Blending with weights: LGB={lgb_weight}, k-NN={knn_weight}")
        merged['price_blended'] = lgb_weight * merged['price'] + knn_weight * merged['knn_price']
        
        # Statistics
        logger.info(f"LightGBM price range: ${merged['price'].min():.2f} - ${merged['price'].max():.2f}")
        logger.info(f"k-NN price range: ${merged['knn_price'].min():.2f} - ${merged['knn_price'].max():.2f}")
        logger.info(f"Blended price range: ${merged['price_blended'].min():.2f} - ${merged['price_blended'].max():.2f}")
        
        # Save blended predictions
        output_df = merged[['sample_id', 'price_blended']].copy()
        output_df.columns = ['sample_id', 'price']
        
    elif 'price_lgb' in merged.columns and 'knn_price' in merged.columns:
        logger.info(f"Blending with weights: LGB={lgb_weight}, k-NN={knn_weight}")
        merged['price'] = lgb_weight * merged['price_lgb'] + knn_weight * merged['knn_price']
        
        # Statistics
        logger.info(f"LightGBM price range: ${merged['price_lgb'].min():.2f} - ${merged['price_lgb'].max():.2f}")
        logger.info(f"k-NN price range: ${merged['knn_price'].min():.2f} - ${merged['knn_price'].max():.2f}")
        logger.info(f"Blended price range: ${merged['price'].min():.2f} - ${merged['price'].max():.2f}")
        
        output_df = merged[['sample_id', 'price']].copy()
    else:
        raise ValueError(f"Unexpected column names: {merged.columns.tolist()}")
    
    output_df.to_csv(output_file, index=False)
    logger.info(f"Blended predictions saved to {output_file}")
    
    # Show sample predictions
    logger.info("\nSample blended predictions:")
    for i in range(min(10, len(output_df))):
        logger.info(f"  {output_df.iloc[i]['sample_id']}: ${output_df.iloc[i]['price']:.2f}")
    
    return output_df

if __name__ == "__main__":
    # Try to find the latest LightGBM prediction file
    import os
    import glob
    
    # Look for prediction files
    pred_files = glob.glob("submissions/test_out*.csv")
    if pred_files:
        lgb_file = sorted(pred_files)[-1]  # Get the latest
        logger.info(f"Using LightGBM file: {lgb_file}")
    else:
        lgb_file = "submissions/test_out.csv"
        logger.info(f"Using default LightGBM file: {lgb_file}")
    
    knn_file = "submissions/knn_preds.csv"
    
    # Check if files exist
    if not os.path.exists(lgb_file):
        logger.error(f"LightGBM file not found: {lgb_file}")
        logger.info("Available files in submissions/:")
        for f in glob.glob("submissions/*.csv"):
            logger.info(f"  {f}")
        exit(1)
    
    if not os.path.exists(knn_file):
        logger.error(f"k-NN file not found: {knn_file}")
        logger.info("Please run knn_price.py first to generate k-NN predictions")
        exit(1)
    
    # Try different weight combinations
    weight_combinations = [
        (0.6, 0.4, "blend_knn_lgb_60_40.csv"),
        (0.5, 0.5, "blend_knn_lgb_50_50.csv"),
        (0.7, 0.3, "blend_knn_lgb_70_30.csv"),
    ]
    
    for lgb_w, knn_w, output_name in weight_combinations:
        logger.info(f"\n{'='*60}")
        logger.info(f"Creating blend with weights LGB={lgb_w}, k-NN={knn_w}")
        logger.info(f"{'='*60}")
        output_file = f"submissions/{output_name}"
        blend_predictions(lgb_file, knn_file, output_file, lgb_w, knn_w)
