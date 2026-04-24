#!/usr/bin/env python3
"""
Feature engineering utilities for product pricing.
Includes IPQ extraction, text preprocessing, and embedding loading.
"""

import os
import re
import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Extract features from product data."""
    
    def __init__(self):
        """Initialize feature extractor with regex patterns."""
        # IPQ (Item Pack Quantity) patterns
        self.ipq_patterns = [
            # Pack of X
            r'(?:pack\s+of\s+|packs\s+of\s+)(\d+)',
            # X pcs/pieces
            r'(\d+)\s*(?:pcs|pieces?|units?)',
            # X x Y format (e.g., "2 x 250ml")
            r'(\d+)\s*x\s*\d+',
            # X count
            r'(\d+)\s*count',
            # X ct
            r'(\d+)\s*ct',
            # Set of X
            r'(?:set\s+of\s+|sets\s+of\s+)(\d+)',
            # Bundle of X
            r'(?:bundle\s+of\s+|bundles\s+of\s+)(\d+)',
            # X-pack
            r'(\d+)-pack',
            # Xpk
            r'(\d+)pk',
            # X ea (each)
            r'(\d+)\s*ea',
            # X each
            r'(\d+)\s*each',
        ]
        
        # Compile regex patterns
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.ipq_patterns]
        
        # Brand keywords (common premium brands)
        self.premium_brands = {
            'apple', 'samsung', 'sony', 'nike', 'adidas', 'gucci', 'louis vuitton', 'chanel',
            'dior', 'hermes', 'prada', 'versace', 'armani', 'calvin klein', 'tommy hilfiger',
            'ralph lauren', 'coach', 'michael kors', 'kate spade', 'tory burch', 'burberry',
            'mont blanc', 'rolex', 'omega', 'cartier', 'tiffany', 'bulgari', 'swarovski',
            'bose', 'bang & olufsen', 'harman kardon', 'marantz', 'denon', 'yamaha',
            'kitchenaid', 'cuisinart', 'vitamix', 'breville', 'wolf', 'sub-zero', 'miele',
            'mercedes', 'bmw', 'audi', 'lexus', 'porsche', 'ferrari', 'lamborghini'
        }
        
        # Category keywords for price estimation
        self.category_keywords = {
            'electronics': ['phone', 'laptop', 'tablet', 'computer', 'tv', 'camera', 'headphones', 'speaker'],
            'clothing': ['shirt', 'dress', 'pants', 'jacket', 'shoes', 'boots', 'sneakers', 'suit'],
            'home': ['furniture', 'chair', 'table', 'sofa', 'bed', 'mattress', 'lamp', 'mirror'],
            'kitchen': ['appliance', 'mixer', 'blender', 'oven', 'refrigerator', 'dishwasher', 'coffee'],
            'beauty': ['makeup', 'skincare', 'perfume', 'shampoo', 'lotion', 'cream', 'serum'],
            'sports': ['equipment', 'gym', 'fitness', 'training', 'outdoor', 'camping', 'hiking'],
            'automotive': ['car', 'tire', 'oil', 'battery', 'brake', 'engine', 'transmission'],
            'jewelry': ['ring', 'necklace', 'bracelet', 'earrings', 'watch', 'pendant', 'chain']
        }
    
    def extract_ipq(self, text: str) -> Optional[int]:
        """
        Extract Item Pack Quantity from text.
        
        Args:
            text: Input text to search
            
        Returns:
            int: Extracted quantity or None if not found
        """
        if not text or pd.isna(text):
            return None
        
        text = str(text).lower()
        
        for pattern in self.compiled_patterns:
            match = pattern.search(text)
            if match:
                try:
                    quantity = int(match.group(1))
                    if 1 <= quantity <= 1000:  # Reasonable range
                        return quantity
                except (ValueError, IndexError):
                    continue
        
        return None
    
    def extract_brand_features(self, text: str) -> Dict[str, bool]:
        """
        Extract brand-related features.
        
        Args:
            text: Input text to search
            
        Returns:
            dict: Brand features
        """
        if not text or pd.isna(text):
            return {'has_premium_brand': False, 'brand_count': 0}
        
        text = str(text).lower()
        found_brands = []
        
        for brand in self.premium_brands:
            if brand in text:
                found_brands.append(brand)
        
        return {
            'has_premium_brand': len(found_brands) > 0,
            'brand_count': len(found_brands)
        }
    
    def extract_category_features(self, text: str) -> Dict[str, bool]:
        """
        Extract category-related features.
        
        Args:
            text: Input text to search
            
        Returns:
            dict: Category features
        """
        if not text or pd.isna(text):
            return {f'category_{cat}': False for cat in self.category_keywords.keys()}
        
        text = str(text).lower()
        category_features = {}
        
        for category, keywords in self.category_keywords.items():
            category_features[f'category_{category}'] = any(keyword in text for keyword in keywords)
        
        return category_features
    
    def extract_text_features(self, text: str) -> Dict[str, float]:
        """
        Extract basic text features.
        
        Args:
            text: Input text
            
        Returns:
            dict: Text features
        """
        if not text or pd.isna(text):
            return {
                'text_length': 0,
                'word_count': 0,
                'char_count': 0,
                'has_numbers': False,
                'has_currency': False,
                'has_dimensions': False
            }
        
        text = str(text)
        
        # Basic text statistics
        word_count = len(text.split())
        char_count = len(text)
        
        # Check for numbers
        has_numbers = bool(re.search(r'\d', text))
        
        # Check for currency symbols
        has_currency = bool(re.search(r'[$€£¥₹]', text))
        
        # Check for dimensions (e.g., "10x20", "5.5 inches")
        has_dimensions = bool(re.search(r'\d+\.?\d*\s*[x×]\s*\d+\.?\d*|\d+\.?\d*\s*(?:inch|cm|mm|ft|feet)', text, re.IGNORECASE))
        
        return {
            'text_length': len(text),
            'word_count': word_count,
            'char_count': char_count,
            'has_numbers': has_numbers,
            'has_currency': has_currency,
            'has_dimensions': has_dimensions
        }
    
    def extract_all_features(self, df: pd.DataFrame, text_col: str = 'catalog_content') -> pd.DataFrame:
        """
        Extract all features from DataFrame.
        
        Args:
            df: Input DataFrame
            text_col: Column name for text content
            
        Returns:
            DataFrame: DataFrame with extracted features
        """
        logger.info(f"Extracting features for {len(df)} samples")
        
        features_df = df.copy()
        
        # Extract IPQ
        features_df['ipq'] = features_df[text_col].apply(self.extract_ipq)
        features_df['has_ipq'] = features_df['ipq'].notna()
        features_df['ipq'] = features_df['ipq'].fillna(1)  # Default to 1 if no IPQ found
        
        # Extract brand features
        brand_features = features_df[text_col].apply(self.extract_brand_features)
        brand_df = pd.DataFrame(brand_features.tolist())
        features_df = pd.concat([features_df, brand_df], axis=1)
        
        # Extract category features
        category_features = features_df[text_col].apply(self.extract_category_features)
        category_df = pd.DataFrame(category_features.tolist())
        features_df = pd.concat([features_df, category_df], axis=1)
        
        # Extract text features
        text_features = features_df[text_col].apply(self.extract_text_features)
        text_df = pd.DataFrame(text_features.tolist())
        features_df = pd.concat([features_df, text_df], axis=1)
        
        logger.info(f"Extracted {len(features_df.columns) - len(df.columns)} additional features")
        
        return features_df

def load_embeddings(embeddings_dir: str, model_name: str = 'optimized') -> Tuple[np.ndarray, Dict]:
    """
    Load cached embeddings and metadata.
    
    Args:
        embeddings_dir: Directory containing embeddings
        model_name: Model name (default: 'optimized')
        
    Returns:
        tuple: (embeddings, metadata)
    """
    embeddings_file = os.path.join(embeddings_dir, 'text_embeddings.npy')
    metadata_file = os.path.join(embeddings_dir, 'metadata.pkl')
    
    if not os.path.exists(embeddings_file):
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
    
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    # Load embeddings
    embeddings = np.load(embeddings_file)
    
    # Load metadata
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
    
    logger.info(f"Loaded {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}")
    
    return embeddings, metadata

def load_image_embeddings(embeddings_dir: str, model_name: str = 'optimized') -> Optional[np.ndarray]:
    """
    Load cached image embeddings.
    
    Args:
        embeddings_dir: Directory containing embeddings
        model_name: Model name (default: 'optimized')
        
    Returns:
        numpy array: Image embeddings or None if not available
    """
    image_embeddings_file = os.path.join(embeddings_dir, 'image_embeddings.npy')
    
    if not os.path.exists(image_embeddings_file):
        logger.warning(f"Image embeddings file not found: {image_embeddings_file}")
        return None
    
    image_embeddings = np.load(image_embeddings_file)
    logger.info(f"Loaded {image_embeddings.shape[0]} image embeddings with dimension {image_embeddings.shape[1]}")
    
    return image_embeddings

def create_feature_matrix(df: pd.DataFrame, embeddings_dir: str, model_name: str = 'optimized') -> np.ndarray:
    """
    Create feature matrix combining text features and embeddings.
    
    Args:
        df: DataFrame with extracted features
        embeddings_dir: Directory containing embeddings
        model_name: Model name (default: 'optimized')
        
    Returns:
        numpy array: Combined feature matrix
    """
    logger.info("Creating feature matrix")
    
    # Load embeddings
    text_embeddings, metadata = load_embeddings(embeddings_dir, model_name)
    image_embeddings = load_image_embeddings(embeddings_dir, model_name)
    
    # Create feature columns list
    feature_columns = []
    
    # Add text embeddings
    for i in range(text_embeddings.shape[1]):
        feature_columns.append(f'text_emb_{i}')
    
    # Add image embeddings if available
    if image_embeddings is not None:
        for i in range(image_embeddings.shape[1]):
            feature_columns.append(f'image_emb_{i}')
    
    # Add engineered features
    engineered_features = [
        'ipq', 'has_ipq', 'has_premium_brand', 'brand_count',
        'text_length', 'word_count', 'char_count', 'has_numbers', 
        'has_currency', 'has_dimensions'
    ]
    
    # Add category features
    category_features = [col for col in df.columns if col.startswith('category_')]
    engineered_features.extend(category_features)
    
    # Add advanced features (TF-IDF and domain-specific)
    advanced_features = [col for col in df.columns if col.startswith('tfidf_') or 
                        col in ['luxury_brand', 'premium_brand', 'mid_range_brand', 'budget_brand',
                               'electronics_category', 'clothing_category', 'home_category', 'kitchen_category',
                               'beauty_category', 'sports_category', 'automotive_category', 'jewelry_category',
                               'premium_quality', 'standard_quality', 'budget_quality',
                               'expensive_material', 'moderate_material', 'cheap_material',
                               'price_keywords', 'quality_keywords', 'size_keywords',
                               'ipq_found', 'ipq_value', 'ipq_log',
                               'positive_sentiment', 'negative_sentiment', 'neutral_sentiment',
                               'technical_specs', 'warranty_mentioned', 'certification_mentioned']]
    
    engineered_features.extend(advanced_features)
    
    # Filter to existing columns
    engineered_features = [col for col in engineered_features if col in df.columns]
    feature_columns.extend(engineered_features)
    
    # Create feature matrix
    feature_matrix = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        sample_id = row['sample_id']
        
        # Find corresponding embedding
        embedding_idx = np.where(metadata['sample_ids'] == sample_id)[0]
        if len(embedding_idx) == 0:
            logger.warning(f"No embedding found for sample {sample_id}")
            continue
        
        embedding_idx = embedding_idx[0]
        
        # Combine features
        features = []
        
        # Add text embedding
        features.extend(text_embeddings[embedding_idx])
        
        # Add image embedding if available
        if image_embeddings is not None:
            features.extend(image_embeddings[embedding_idx])
        
        # Add engineered features
        for col in engineered_features:
            if col in df.columns:
                features.append(row[col])
            else:
                features.append(0)  # Default value
        
        feature_matrix.append(features)
        valid_indices.append(idx)
    
    feature_matrix = np.array(feature_matrix)
    logger.info(f"Created feature matrix with shape {feature_matrix.shape}")
    logger.info(f"Valid samples: {len(valid_indices)}/{len(df)}")
    
    return feature_matrix, feature_columns, valid_indices

def main():
    """Test feature extraction."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test feature extraction')
    parser.add_argument('--csv', required=True, help='Input CSV file')
    parser.add_argument('--embeddings_dir', required=True, help='Embeddings directory')
    parser.add_argument('--model', default='clip', help='Model name')
    
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.csv)
    logger.info(f"Loaded {len(df)} samples")
    
    # Extract features
    extractor = FeatureExtractor()
    features_df = extractor.extract_all_features(df)
    
    # Create feature matrix
    feature_matrix, feature_columns, valid_indices = create_feature_matrix(features_df, args.embeddings_dir, args.model)
    
    logger.info(f"Feature matrix shape: {feature_matrix.shape}")
    logger.info(f"Feature columns: {len(feature_columns)}")
    logger.info(f"Valid indices: {len(valid_indices)}")

if __name__ == "__main__":
    main()
