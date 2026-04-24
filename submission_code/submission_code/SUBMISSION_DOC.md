# Amazon ML Challenge 2025 - Solution Documentation

## Team Information
- **Team Name**: Synergy
- **Team Members**: Shaikh Zaid , Karan Jagtap , Riaan Atar

## 1. Solution Overview
A multi-modal approach combining text and image features using LightGBM for product price prediction.

## 2. Methodology

### Data Processing
- **Text**: Cleaned catalog content, extracted IPQ, brand names
- **Images**: Processed using CLIP, handled missing images with zero vectors

### Feature Engineering
- **Text Features**:
  - MiniLM-L6-v2 embeddings (384D)
  - Character/word counts, brand presence
- **Image Features**:
  - CLIP-ViT-Base embeddings (512D)

### Model
- **Algorithm**: LightGBM with log1p target transform
- **Key Params**: learning_rate=0.01, num_leaves=255
- **CV**: 5-fold, SMAPE optimization

## 3. Results
- **CV SMAPE**: 21.63%
- **CV MAE**: 0.5490
- **CV R²**: 0.4367

## 4. Dependencies
- Python 3.8+, PyTorch, LightGBM, Transformers