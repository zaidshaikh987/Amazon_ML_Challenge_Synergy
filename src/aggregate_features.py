import pandas as pd
import numpy as np
import re
from collections import defaultdict

class AggregateFeatureExtractor:
    """
    Extract aggregate statistics (median, mean, std) for categories, brands, etc.
    These are powerful features that capture price ranges for different product types.
    """
    
    def __init__(self):
        self.category_stats = {}
        self.ipq_stats = {}
        self.brand_stats = {}
        self.combined_stats = {}
        self.global_median = 0
        
    def extract_brand(self, catalog_content):
        """Extract potential brand from catalog_content (first word of Item Name)."""
        if pd.isna(catalog_content):
            return "unknown"
        # Extract Item Name from catalog_content
        try:
            lines = str(catalog_content).split('\n')
            for line in lines:
                if 'Item Name:' in line:
                    title = line.replace('Item Name:', '').strip()
                    words = title.split()
                    if len(words) > 0:
                        return words[0].lower()
        except:
            pass
        return "unknown"
    
    def extract_category_hint(self, catalog_content):
        """Extract category hints from catalog_content."""
        if pd.isna(catalog_content):
            return "unknown"
        
        # Extract full text from catalog_content
        catalog_lower = str(catalog_content).lower()
        
        # Common product categories
        categories = {
            'electronics': ['phone', 'laptop', 'computer', 'tablet', 'camera', 'headphone', 'speaker', 'tv', 'monitor'],
            'clothing': ['shirt', 'pant', 'dress', 'shoe', 'jacket', 'coat', 'jeans', 'sweater'],
            'home': ['furniture', 'table', 'chair', 'bed', 'lamp', 'sofa', 'desk'],
            'kitchen': ['pan', 'pot', 'dish', 'cup', 'plate', 'knife', 'appliance'],
            'beauty': ['makeup', 'cosmetic', 'perfume', 'lotion', 'cream', 'shampoo'],
            'toys': ['toy', 'game', 'puzzle', 'doll', 'lego', 'action figure'],
            'sports': ['fitness', 'exercise', 'yoga', 'dumbbell', 'bike', 'ball'],
            'books': ['book', 'novel', 'textbook', 'guide', 'manual'],
        }
        
        for category, keywords in categories.items():
            for keyword in keywords:
                if keyword in catalog_lower:
                    return category
        
        return "other"
    
    def fit(self, df):
        """
        Fit aggregate statistics on training data.
        
        Args:
            df: DataFrame with 'catalog_content', 'price' columns
        """
        print("Extracting aggregate features from training data...")
        
        # Extract brand and category from catalog_content
        df = df.copy()
        df['brand'] = df['catalog_content'].apply(self.extract_brand)
        df['category_hint'] = df['catalog_content'].apply(self.extract_category_hint)
        
        # Create synthetic product type ID from category (since we don't have PRODUCT_TYPE_ID)
        df['product_type'] = df['category_hint']
        
        # Global median
        self.global_median = df['price'].median()
        
        # Category statistics
        print("  Computing category statistics...")
        category_grouped = df.groupby('category_hint')['price'].agg(['median', 'mean', 'std', 'count'])
        self.category_stats = category_grouped.to_dict('index')
        
        # Product type statistics
        print("  Computing product type statistics...")
        ipq_grouped = df.groupby('product_type')['price'].agg(['median', 'mean', 'std', 'count'])
        self.ipq_stats = ipq_grouped.to_dict('index')
        
        # Brand statistics
        print("  Computing brand statistics...")
        brand_grouped = df.groupby('brand')['price'].agg(['median', 'mean', 'std', 'count'])
        # Only keep brands with at least 5 occurrences
        brand_grouped = brand_grouped[brand_grouped['count'] >= 5]
        self.brand_stats = brand_grouped.to_dict('index')
        
        # Combined: category + product_type (just use category for now)
        print("  Computing combined statistics...")
        # Since product_type is same as category, skip combined
        self.combined_stats = {}
        
        print(f"  Unique categories: {len(self.category_stats)}")
        print(f"  Unique product types: {len(self.ipq_stats)}")
        print(f"  Unique brands: {len(self.brand_stats)}")
        print(f"  Unique combinations: {len(self.combined_stats)}")
        
        return self
    
    def transform(self, df):
        """
        Transform DataFrame by adding aggregate features.
        
        Args:
            df: DataFrame with 'catalog_content' column
        
        Returns:
            DataFrame with additional aggregate feature columns
        """
        df = df.copy()
        
        # Extract brand and category from catalog_content
        df['brand'] = df['catalog_content'].apply(self.extract_brand)
        df['category_hint'] = df['catalog_content'].apply(self.extract_category_hint)
        df['product_type'] = df['category_hint']
        
        # Category aggregate features
        df['category_median_price'] = df['category_hint'].map(
            lambda x: self.category_stats.get(x, {}).get('median', self.global_median)
        )
        df['category_mean_price'] = df['category_hint'].map(
            lambda x: self.category_stats.get(x, {}).get('mean', self.global_median)
        )
        df['category_std_price'] = df['category_hint'].map(
            lambda x: self.category_stats.get(x, {}).get('std', 0)
        )
        
        # Product type aggregate features
        df['product_type_median_price'] = df['product_type'].map(
            lambda x: self.ipq_stats.get(x, {}).get('median', self.global_median)
        )
        df['product_type_mean_price'] = df['product_type'].map(
            lambda x: self.ipq_stats.get(x, {}).get('mean', self.global_median)
        )
        df['product_type_std_price'] = df['product_type'].map(
            lambda x: self.ipq_stats.get(x, {}).get('std', 0)
        )
        
        # Brand aggregate features
        df['brand_median_price'] = df['brand'].map(
            lambda x: self.brand_stats.get(x, {}).get('median', self.global_median)
        )
        df['brand_mean_price'] = df['brand'].map(
            lambda x: self.brand_stats.get(x, {}).get('mean', self.global_median)
        )
        
        # Drop temporary columns
        df = df.drop(['brand', 'category_hint', 'product_type'], axis=1)
        
        print(f"Added aggregate features. New shape: {df.shape}")
        
        return df
    
    def fit_transform(self, df):
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df)


def add_aggregate_features_to_dataset():
    """
    Standalone function to add aggregate features to train/test CSVs.
    """
    print("="*70)
    print("ADDING AGGREGATE FEATURES")
    print("="*70)
    
    # Load data
    print("\n1. Loading data...")
    train = pd.read_csv("student_resource/dataset/train.csv")
    test = pd.read_csv("student_resource/dataset/test.csv")
    
    print(f"   Train: {train.shape}")
    print(f"   Test: {test.shape}")
    
    # Extract features
    print("\n2. Extracting aggregate features...")
    extractor = AggregateFeatureExtractor()
    
    # Fit on train
    extractor.fit(train)
    
    # Transform both
    train_agg = extractor.transform(train)
    test_agg = extractor.transform(test)
    
    # Save
    print("\n3. Saving enriched datasets...")
    train_agg.to_csv("student_resource/dataset/train_with_aggregates.csv", index=False)
    test_agg.to_csv("student_resource/dataset/test_with_aggregates.csv", index=False)
    
    print("   Saved to:")
    print("     train_with_aggregates.csv")
    print("     test_with_aggregates.csv")
    
    print("\n4. Sample of new features:")
    new_cols = [c for c in train_agg.columns if 'price' in c.lower() and c != 'price']
    print(train_agg[new_cols].head())
    
    print("\n" + "="*70)
    print("DONE! Use these files for training with aggregate features.")
    print("="*70)
    
    return train_agg, test_agg


if __name__ == "__main__":
    add_aggregate_features_to_dataset()
