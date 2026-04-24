#!/usr/bin/env python3
"""
Optimized embeddings extractor using sentence-transformers/all-MiniLM-L12-v2 + CLIP.
Extracts text and image embeddings efficiently with caching.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
from tqdm import tqdm
import requests
from PIL import Image
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import models with fallback
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    logger.info("sentence-transformers available")
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available, using fallback")

try:
    from transformers import CLIPModel, CLIPProcessor
    import torch
    TRANSFORMERS_AVAILABLE = True
    logger.info("transformers available")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not available, using fallback")

class OptimizedEmbeddingsExtractor:
    """Optimized embeddings extractor using best models."""
    
    def __init__(self):
        """Initialize optimized embeddings extractor."""
        self.text_model = None
        self.image_model = None
        self.image_processor = None
        
        # Check GPU availability
        self.using_gpu = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.using_gpu else 'cpu')
        
        logger.info(f"Using device: {self.device}")
        if self.using_gpu:
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            logger.info("No GPU detected, using CPU")
        
        # Initialize text model (FASTER MODEL!)
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # Use faster, smaller model for better performance
                self.text_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # Smaller than L12
                logger.info(f"Loaded sentence-transformers/all-MiniLM-L6-v2 (faster)")
            except Exception as e:
                logger.warning(f"Failed to load faster model, trying original: {str(e)}")
                try:
                    self.text_model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
                    logger.info("Loaded sentence-transformers/all-MiniLM-L12-v2")
                except Exception as e2:
                    logger.warning(f"Failed to load sentence-transformers model: {str(e2)}")
                    self.text_model = None
        
        # Initialize image model
        if TRANSFORMERS_AVAILABLE:
            try:
                self.image_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self.image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                
                # Move to GPU if available
                if self.using_gpu:
                    self.image_model = self.image_model.to(self.device)
                
                logger.info(f"Loaded openai/clip-vit-base-patch32 on {self.device}")
            except Exception as e:
                logger.warning(f"Failed to load CLIP model: {str(e)}")
                self.image_model = None
                self.image_processor = None
    
    def extract_text_embedding(self, text: str) -> np.ndarray:
        """Extract text embedding using sentence-transformers."""
        if self.text_model is not None:
            try:
                embedding = self.text_model.encode([text], show_progress_bar=False)
                return embedding[0]
            except Exception as e:
                logger.warning(f"Failed to extract text embedding: {str(e)}")
                return self._extract_fallback_text_features(text)
        else:
            return self._extract_fallback_text_features(text)
    
    def extract_image_embedding(self, image_path: str) -> np.ndarray:
        """Extract image embedding using CLIP."""
        if self.image_model is not None and self.image_processor is not None:
            try:
                # Load image
                image = Image.open(image_path).convert("RGB")
                
                # Process image
                inputs = self.image_processor(images=image, return_tensors="pt")
                
                # Move inputs to GPU if available
                if self.using_gpu:
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get embedding
                with torch.no_grad():
                    embedding = self.image_model.get_image_features(**inputs)
                
                return embedding.squeeze().cpu().numpy()
            except Exception as e:
                logger.warning(f"Failed to extract image embedding from {image_path}: {str(e)}")
                return self._extract_fallback_image_features(image_path)
        else:
            return self._extract_fallback_image_features(image_path)
    
    def _extract_fallback_text_features(self, text: str) -> np.ndarray:
        """Extract fallback text features."""
        if not text or pd.isna(text):
            return np.zeros(384)  # Match sentence-transformers dimension
        
        text = str(text).lower()
        
        # Basic text statistics
        features = []
        
        # Length features
        features.extend([
            len(text),
            len(text.split()),
            len(text.replace(' ', '')),
            len(set(text.split())),  # unique words
        ])
        
        # Character type features
        features.extend([
            sum(1 for c in text if c.isalpha()),
            sum(1 for c in text if c.isdigit()),
            sum(1 for c in text if c.isspace()),
            sum(1 for c in text if c.isalnum()),
        ])
        
        # Keyword features
        keywords = ['premium', 'luxury', 'quality', 'professional', 'high-end', 'brand', 'original', 'authentic']
        features.extend([sum(1 for kw in keywords if kw in text)])
        
        # Price-related features
        price_keywords = ['price', 'cost', 'expensive', 'cheap', 'sale', 'discount', 'offer', 'deal']
        features.extend([sum(1 for kw in price_keywords if kw in text)])
        
        # Size/quantity features
        size_keywords = ['large', 'small', 'big', 'tiny', 'huge', 'mini', 'jumbo', 'pack', 'set', 'bundle']
        features.extend([sum(1 for kw in size_keywords if kw in text)])
        
        # Pad or truncate to target dimension
        target_dim = 384
        if len(features) < target_dim:
            features.extend([0] * (target_dim - len(features)))
        else:
            features = features[:target_dim]
        
        return np.array(features, dtype=np.float32)
    
    def _extract_fallback_image_features(self, image_path: str) -> np.ndarray:
        """Extract fallback image features."""
        try:
            from PIL import Image
            import numpy as np
            
            image = Image.open(image_path).convert('RGB')
            img_array = np.array(image)
            
            # Basic image statistics
            features = []
            
            # Shape features
            features.extend([
                img_array.shape[0],  # height
                img_array.shape[1],  # width
                img_array.shape[0] * img_array.shape[1],  # total pixels
                img_array.shape[0] / img_array.shape[1] if img_array.shape[1] > 0 else 0,  # aspect ratio
            ])
            
            # Color statistics
            features.extend([
                np.mean(img_array),
                np.std(img_array),
                np.min(img_array),
                np.max(img_array),
                np.median(img_array),
            ])
            
            # Per-channel statistics
            for channel in range(3):
                channel_data = img_array[:, :, channel]
                features.extend([
                    np.mean(channel_data),
                    np.std(channel_data),
                    np.min(channel_data),
                    np.max(channel_data),
                ])
            
            # Pad or truncate to target dimension
            target_dim = 512  # Match CLIP dimension
            if len(features) < target_dim:
                features.extend([0] * (target_dim - len(features)))
            else:
                features = features[:target_dim]
            
            return np.array(features, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Failed to extract fallback image features from {image_path}: {str(e)}")
            return np.zeros(512)  # Match CLIP dimension
    
    def extract_embeddings_batch(self, df: pd.DataFrame, image_dir: str, 
                                text_col: str = 'catalog_content', 
                                id_col: str = 'sample_id',
                                batch_size: int = 32) -> tuple:  # Smaller batches for faster individual processing
        """
        Extract embeddings for a batch of samples (OPTIMIZED with batching).
        
        Args:
            df: DataFrame with text and image paths
            image_dir: Directory containing images
            text_col: Text column name
            id_col: ID column name
            batch_size: Batch size for processing (default: 32 for faster individual batches)
            
        Returns:
            tuple: (text embeddings, image embeddings, metadata)
        """
        logger.info(f"Extracting embeddings for {len(df)} samples with batch_size={batch_size}")
        
        sample_ids = df[id_col].tolist()
        texts = [row[text_col] if pd.notna(row[text_col]) else "" for _, row in df.iterrows()]
        
        # BATCH TEXT EMBEDDINGS (OPTIMIZED!)
        logger.info("Extracting text embeddings in batches...")
        text_embeddings = []
        if self.text_model is not None:
            total_batches = (len(texts) + batch_size - 1) // batch_size
            for i in tqdm(range(0, len(texts), batch_size), desc="Text embeddings", total=total_batches):
                batch_texts = texts[i:i+batch_size]
                batch_embs = self.text_model.encode(
                    batch_texts, 
                    show_progress_bar=False, 
                    batch_size=batch_size,
                    convert_to_numpy=True  # More efficient
                )
                text_embeddings.extend(batch_embs)
        else:
            # Fallback
            for text in tqdm(texts, desc="Text embeddings (fallback)"):
                text_embeddings.append(self._extract_fallback_text_features(text))
        
        # BATCH IMAGE EMBEDDINGS (OPTIMIZED!)
        logger.info("Extracting image embeddings in batches...")
        image_embeddings = []
        if self.image_model is not None and self.image_processor is not None:
            total_batches = (len(sample_ids) + batch_size - 1) // batch_size
            for i in tqdm(range(0, len(sample_ids), batch_size), desc="Image embeddings", total=total_batches):
                batch_ids = sample_ids[i:i+batch_size]
                batch_images = []
                
                for sample_id in batch_ids:
                    image_path = os.path.join(image_dir, f"{sample_id}.jpg")
                    try:
                        if os.path.exists(image_path):
                            image = Image.open(image_path).convert("RGB")
                            batch_images.append(image)
                        else:
                            # Use black image as placeholder
                            batch_images.append(Image.new('RGB', (224, 224), (0, 0, 0)))
                    except:
                        batch_images.append(Image.new('RGB', (224, 224), (0, 0, 0)))
                
                # Process batch
                inputs = self.image_processor(images=batch_images, return_tensors="pt")
                
                # Move inputs to GPU if available
                if self.using_gpu:
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    batch_embs = self.image_model.get_image_features(**inputs)
                image_embeddings.extend(batch_embs.cpu().numpy())
        else:
            # Fallback
            for sample_id in tqdm(sample_ids, desc="Image embeddings (fallback)"):
                image_path = os.path.join(image_dir, f"{sample_id}.jpg")
                image_embeddings.append(self._extract_fallback_image_features(image_path))
        
        text_embeddings = np.array(text_embeddings)
        image_embeddings = np.array(image_embeddings)
        
        metadata = {
            'sample_ids': np.array(sample_ids),
            'text_dim': text_embeddings.shape[1],
            'image_dim': image_embeddings.shape[1],
            'device': str(self.device),
            'using_gpu': self.using_gpu,
            'model_info': {
                'text_model': 'sentence-transformers/all-MiniLM-L6-v2' if self.text_model else 'fallback',
                'image_model': 'openai/clip-vit-base-patch32' if self.image_model else 'fallback'
            }
        }
        
        logger.info(f"Extracted embeddings:")
        logger.info(f"  Text: {text_embeddings.shape}")
        logger.info(f"  Image: {image_embeddings.shape}")
        
        return text_embeddings, image_embeddings, metadata

def main():
    parser = argparse.ArgumentParser(description='Extract optimized embeddings')
    parser.add_argument('--csv', required=True, help='Input CSV file')
    parser.add_argument('--image_dir', required=True, help='Image directory')
    parser.add_argument('--out_dir', required=True, help='Output directory')
    parser.add_argument('--text_col', default='catalog_content', help='Text column name')
    parser.add_argument('--id_col', default='sample_id', help='ID column name')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.csv):
        logger.error(f"CSV file not found: {args.csv}")
        sys.exit(1)
    
    if not os.path.exists(args.image_dir):
        logger.error(f"Image directory not found: {args.image_dir}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    try:
        # Load data
        logger.info(f"Loading data from {args.csv}")
        df = pd.read_csv(args.csv)
        logger.info(f"Loaded {len(df)} samples")
        
        # Check required columns
        required_cols = [args.id_col, args.text_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            sys.exit(1)
        
        # Extract embeddings
        extractor = OptimizedEmbeddingsExtractor()
        text_embeddings, image_embeddings, metadata = extractor.extract_embeddings_batch(
            df, args.image_dir, args.text_col, args.id_col
        )
        
        # Save embeddings
        text_embeddings_file = os.path.join(args.out_dir, 'text_embeddings.npy')
        image_embeddings_file = os.path.join(args.out_dir, 'image_embeddings.npy')
        metadata_file = os.path.join(args.out_dir, 'metadata.pkl')
        
        np.save(text_embeddings_file, text_embeddings)
        np.save(image_embeddings_file, image_embeddings)
        
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Embeddings saved:")
        logger.info(f"  Text: {text_embeddings_file}")
        logger.info(f"  Image: {image_embeddings_file}")
        logger.info(f"  Metadata: {metadata_file}")
        
        # Print model info
        logger.info("Model information:")
        logger.info(f"  Text model: {metadata['model_info']['text_model']}")
        logger.info(f"  Image model: {metadata['model_info']['image_model']}")
        logger.info(f"  Text dimension: {metadata['text_dim']}")
        logger.info(f"  Image dimension: {metadata['image_dim']}")
        
        logger.info("Embedding extraction completed successfully!")
        
    except Exception as e:
        logger.error(f"Embedding extraction failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()