#!/usr/bin/env python3
"""
k-NN price predictor using HNSW for fast approximate nearest neighbor search.
"""
import numpy as np
import pandas as pd
import logging
import gc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import hnswlib
    logger.info("hnswlib is available")
except ImportError:
    logger.error("hnswlib not found. Installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'hnswlib'])
    import hnswlib
    logger.info("hnswlib installed successfully")

def main():
    logger.info("Loading training embeddings...")
    X_img = np.load("embeddings/train/image_embeddings.npy")   # (N,512)
    X_txt = np.load("embeddings/train/text_embeddings.npy")    # (N,384)
    logger.info(f"Train image embeddings shape: {X_img.shape}")
    logger.info(f"Train text embeddings shape: {X_txt.shape}")
    
    # Concatenate embeddings with memory management
    logger.info("Concatenating embeddings...")
    X = np.hstack([X_img, X_txt]).astype('float32')
    logger.info(f"Combined train embeddings shape: {X.shape}")
    
    # Clean up individual arrays to save memory
    del X_img, X_txt
    gc.collect()
    
    # Load train prices
    logger.info("Loading training prices...")
    train_df = pd.read_csv("student_resource/dataset/train.csv")
    y = train_df['price'].values
    logger.info(f"Train prices shape: {y.shape}")
    logger.info(f"Price range: ${y.min():.2f} - ${y.max():.2f}")
    
    # Build HNSW index with more conservative parameters
    logger.info("Building HNSW index...")
    dim = X.shape[1]
    logger.info(f"Index dimensions: {dim}")
    
    # Use more conservative HNSW parameters for stability
    p = hnswlib.Index(space='l2', dim=dim)
    max_elements = X.shape[0]
    logger.info(f"Max elements: {max_elements}")
    
    # Reduce construction parameters to avoid memory issues
    p.init_index(max_elements=max_elements, ef_construction=100, M=16)
    logger.info("HNSW index initialized")
    
    # Add items in smaller batches to avoid memory spikes
    batch_size = 10000
    for i in range(0, max_elements, batch_size):
        end_idx = min(i + batch_size, max_elements)
        logger.info(f"Adding items {i} to {end_idx}...")
        p.add_items(X[i:end_idx], np.arange(i, end_idx))
    
    # Set search parameters
    p.set_ef(50)  # Lower search parameter for speed
    logger.info("HNSW index built successfully")
    
    # Clean up training data
    del X
    gc.collect()
    
    # Load test embeddings
    logger.info("Loading test embeddings...")
    X_test_img = np.load("embeddings/test/image_embeddings.npy")
    X_test_txt = np.load("embeddings/test/text_embeddings.npy")
    logger.info(f"Test image embeddings shape: {X_test_img.shape}")
    logger.info(f"Test text embeddings shape: {X_test_txt.shape}")
    
    X_test = np.hstack([X_test_img, X_test_txt]).astype('float32')
    logger.info(f"Combined test embeddings shape: {X_test.shape}")
    
    # Clean up individual test arrays
    del X_test_img, X_test_txt
    gc.collect()
    
    # Query test set in batches
    logger.info("Finding k-nearest neighbors (k=5)...")
    k = 5
    batch_size = 1000
    
    all_labels = []
    all_distances = []
    
    for i in range(0, X_test.shape[0], batch_size):
        end_idx = min(i + batch_size, X_test.shape[0])
        batch = X_test[i:end_idx]
        logger.info(f"Processing test batch {i} to {end_idx}...")
        
        labels, distances = p.knn_query(batch, k=k)
        all_labels.append(labels)
        all_distances.append(distances)
    
    # Combine results
    labels = np.vstack(all_labels)
    distances = np.vstack(all_distances)
    logger.info(f"k-NN search completed")
    
    # Median price of neighbors
    logger.info("Computing median prices from neighbors...")
    knn_preds = np.median(y[labels], axis=1)
    logger.info(f"k-NN predictions shape: {knn_preds.shape}")
    logger.info(f"k-NN predictions range: ${knn_preds.min():.2f} - ${knn_preds.max():.2f}")
    logger.info(f"k-NN predictions mean: ${knn_preds.mean():.2f}")
    logger.info(f"k-NN predictions median: ${np.median(knn_preds):.2f}")
    
    # Save predictions
    test_df = pd.read_csv("student_resource/dataset/test.csv")
    output_df = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'knn_price': knn_preds
    })
    
    output_path = "submissions/knn_preds.csv"
    output_df.to_csv(output_path, index=False)
    logger.info(f"k-NN predictions saved to {output_path}")
    
    # Show sample predictions
    logger.info("\nSample k-NN predictions:")
    for i in range(min(10, len(output_df))):
        logger.info(f"  {output_df.iloc[i]['sample_id']}: ${output_df.iloc[i]['knn_price']:.2f}")

if __name__ == "__main__":
    main()
