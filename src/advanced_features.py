#!/usr/bin/env python3
"""
Advanced feature engineering to get below 40% SMAPE.
Extracts domain-specific features: brand, material, size, quantity, etc.
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

class AdvancedFeatureExtractor:
    """Extract advanced domain-specific features."""
    
    def __init__(self):
        # Premium brand detection (expanded)
        self.premium_brands = {
            'electronics': ['apple', 'samsung', 'sony', 'lg', 'bose', 'canon', 'nikon', 'gopro', 'dyson'],
            'fashion': ['nike', 'adidas', 'puma', 'gucci', 'prada', 'versace', 'armani', 'burberry', 'chanel'],
            'beauty': ['loreal', 'olay', 'estee lauder', 'clinique', 'lancome', 'mac', 'nars'],
            'home': ['kitchenaid', 'cuisinart', 'vitamix', 'breville', 'le creuset', 'lodge']
        }
        
        # Mid-range brands
        self.midrange_brands = {
            'electronics': ['hp', 'dell', 'lenovo', 'asus', 'acer', 'philips', 'panasonic'],
            'fashion': ['levi', 'tommy', 'calvin', 'polo', 'gap', 'zara', 'h&m'],
            'home': ['oxo', 'pyrex', 'rubbermaid', 'tupperware']
        }
        
        # Materials (affect price significantly)
        self.materials = {
            'premium': ['gold', 'silver', 'platinum', 'diamond', 'leather', 'silk', 'cashmere', 
                       'stainless steel', 'titanium', 'carbon fiber', 'wool', 'suede'],
            'standard': ['cotton', 'polyester', 'plastic', 'aluminum', 'wood', 'glass', 'ceramic'],
            'synthetic': ['nylon', 'acrylic', 'pvc', 'vinyl', 'rubber']
        }
        
        # Size indicators (larger often = more expensive)
        self.size_keywords = {
            'large': ['large', 'xl', 'xxl', 'big', 'jumbo', 'king', 'queen'],
            'medium': ['medium', 'regular', 'standard'],
            'small': ['small', 'mini', 'compact', 'travel', 'pocket']
        }
        
        # Quantity patterns (more aggressive)
        self.quantity_patterns = [
            r'(\d+)\s*(?:pack|pk|pcs|pieces|units|count|ct)',
            r'pack\s*of\s*(\d+)',
            r'set\s*of\s*(\d+)',
            r'(\d+)\s*x\s*\d+',  # 2 x 500ml
            r'(\d+)-pack',
            r'(\d+)pk',
            r'box\s*of\s*(\d+)',
            r'case\s*of\s*(\d+)',
        ]
        
        # Weight/volume indicators
        self.weight_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:kg|kilogram|pound|lb|lbs|oz|ounce)',
            r'(\d+(?:\.\d+)?)\s*(?:ml|liter|litre|gallon|gal|fl oz)',
        ]
        
        # Product categories (price ranges vary)
        self.categories = {
            'high_value': ['laptop', 'computer', 'tv', 'camera', 'phone', 'tablet', 'watch', 
                          'furniture', 'mattress', 'refrigerator', 'washing machine'],
            'mid_value': ['headphone', 'speaker', 'keyboard', 'mouse', 'monitor', 'printer',
                         'chair', 'desk', 'lamp', 'blender', 'toaster'],
            'low_value': ['cable', 'charger', 'case', 'cover', 'adapter', 'accessory',
                         'pen', 'pencil', 'notebook', 'bag', 'bottle']
        }
        
        # Condition keywords
        self.condition_keywords = {
            'new': ['new', 'brand new', 'factory sealed', 'unopened'],
            'refurbished': ['refurbished', 'renewed', 'certified', 'reconditioned'],
            'used': ['used', 'pre-owned', 'second hand', 'open box']
        }
    
    def extract_brand_tier(self, text: str) -> Dict[str, float]:
        """Detect brand tier (premium/mid/budget)."""
        text_lower = text.lower()
        
        # Check premium brands
        for category, brands in self.premium_brands.items():
            for brand in brands:
                if brand in text_lower:
                    return {'brand_tier': 3.0, f'brand_cat_{category}': 1.0}
        
        # Check mid-range brands
        for category, brands in self.midrange_brands.items():
            for brand in brands:
                if brand in text_lower:
                    return {'brand_tier': 2.0, f'brand_cat_{category}': 1.0}
        
        # Default: budget/unknown
        return {'brand_tier': 1.0}
    
    def extract_material_quality(self, text: str) -> Dict[str, float]:
        """Detect material type (premium/standard/synthetic)."""
        text_lower = text.lower()
        
        features = {
            'material_premium': 0.0,
            'material_standard': 0.0,
            'material_synthetic': 0.0,
            'material_count': 0.0
        }
        
        for material in self.materials['premium']:
            if material in text_lower:
                features['material_premium'] = 1.0
                features['material_count'] += 1.0
        
        for material in self.materials['standard']:
            if material in text_lower:
                features['material_standard'] = 1.0
                features['material_count'] += 1.0
        
        for material in self.materials['synthetic']:
            if material in text_lower:
                features['material_synthetic'] = 1.0
                features['material_count'] += 1.0
        
        return features
    
    def extract_size_category(self, text: str) -> Dict[str, float]:
        """Detect size category."""
        text_lower = text.lower()
        
        for size_cat, keywords in self.size_keywords.items():
            for kw in keywords:
                if kw in text_lower:
                    return {f'size_{size_cat}': 1.0}
        
        return {'size_medium': 0.5}  # Default
    
    def extract_quantity(self, text: str) -> Dict[str, float]:
        """Extract quantity/pack size."""
        text_lower = text.lower()
        
        for pattern in self.quantity_patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    qty = int(match.group(1))
                    return {
                        'quantity': float(qty),
                        'quantity_log': np.log1p(qty),
                        'is_multipack': 1.0 if qty > 1 else 0.0
                    }
                except:
                    pass
        
        return {'quantity': 1.0, 'quantity_log': 0.0, 'is_multipack': 0.0}
    
    def extract_weight_volume(self, text: str) -> Dict[str, float]:
        """Extract weight/volume."""
        features = {'has_weight': 0.0, 'weight_value': 0.0}
        
        for pattern in self.weight_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    features['has_weight'] = 1.0
                    features['weight_value'] = np.log1p(value)
                    return features
                except:
                    pass
        
        return features
    
    def extract_category_value(self, text: str) -> Dict[str, float]:
        """Categorize by typical price range."""
        text_lower = text.lower()
        
        for kw in self.categories['high_value']:
            if kw in text_lower:
                return {'category_value': 3.0, 'is_high_value': 1.0}
        
        for kw in self.categories['mid_value']:
            if kw in text_lower:
                return {'category_value': 2.0, 'is_mid_value': 1.0}
        
        for kw in self.categories['low_value']:
            if kw in text_lower:
                return {'category_value': 1.0, 'is_low_value': 1.0}
        
        return {'category_value': 1.5}  # Default
    
    def extract_condition(self, text: str) -> Dict[str, float]:
        """Extract condition."""
        text_lower = text.lower()
        
        for condition, keywords in self.condition_keywords.items():
            for kw in keywords:
                if kw in text_lower:
                    value = {'new': 1.0, 'refurbished': 0.7, 'used': 0.3}.get(condition, 0.5)
                    return {f'condition_{condition}': 1.0, 'condition_value': value}
        
        return {'condition_new': 0.5, 'condition_value': 0.8}  # Assume new by default
    
    def extract_numeric_features(self, text: str) -> Dict[str, float]:
        """Extract numeric features from text."""
        # Count numbers (often indicates specs)
        numbers = re.findall(r'\d+', text)
        
        return {
            'number_count': float(len(numbers)),
            'has_large_numbers': 1.0 if any(int(n) > 1000 for n in numbers if len(n) <= 6) else 0.0,
            'max_number': float(max([int(n) for n in numbers if len(n) <= 6], default=0))
        }
    
    def extract_all_advanced_features(self, text: str) -> Dict[str, float]:
        """Extract all advanced features."""
        if pd.isna(text) or not text:
            text = ""
        
        features = {}
        
        # Combine all feature extractors
        features.update(self.extract_brand_tier(text))
        features.update(self.extract_material_quality(text))
        features.update(self.extract_size_category(text))
        features.update(self.extract_quantity(text))
        features.update(self.extract_weight_volume(text))
        features.update(self.extract_category_value(text))
        features.update(self.extract_condition(text))
        features.update(self.extract_numeric_features(text))
        
        # Fill missing keys with 0
        expected_keys = [
            'brand_tier', 'material_premium', 'material_standard', 'material_synthetic',
            'material_count', 'size_large', 'size_medium', 'size_small',
            'quantity', 'quantity_log', 'is_multipack', 'has_weight', 'weight_value',
            'category_value', 'is_high_value', 'is_mid_value', 'is_low_value',
            'condition_new', 'condition_refurbished', 'condition_used', 'condition_value',
            'number_count', 'has_large_numbers', 'max_number'
        ]
        
        for key in expected_keys:
            if key not in features:
                features[key] = 0.0
        
        return features

def add_advanced_features_to_df(df: pd.DataFrame, text_col: str = 'catalog_content') -> pd.DataFrame:
    """Add advanced features to dataframe."""
    extractor = AdvancedFeatureExtractor()
    
    print(f"Extracting advanced features from {len(df)} samples...")
    
    advanced_features = []
    for text in df[text_col]:
        features = extractor.extract_all_advanced_features(text)
        advanced_features.append(features)
    
    # Convert to DataFrame
    advanced_df = pd.DataFrame(advanced_features)
    
    # Combine with original
    result_df = pd.concat([df.reset_index(drop=True), advanced_df.reset_index(drop=True)], axis=1)
    
    print(f"Added {len(advanced_df.columns)} advanced features")
    
    return result_df

if __name__ == "__main__":
    # Test
    test_text = "Apple iPhone 14 Pro Max 256GB - Brand New Factory Sealed - Premium Leather Case Included"
    extractor = AdvancedFeatureExtractor()
    features = extractor.extract_all_advanced_features(test_text)
    
    print("Test features:")
    for k, v in sorted(features.items()):
        if v != 0:
            print(f"  {k}: {v}")
