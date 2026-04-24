#!/bin/bash
# Quick end-to-end test on sample data
# This script runs the complete pipeline on sample_test.csv

set -e  # Exit on any error

echo "=== Amazon ML Challenge 2025 - Sample Pipeline Test ==="
echo "Starting end-to-end test on sample data..."
echo

# Check if we're in the right directory
if [ ! -f "student_resource/dataset/sample_test.csv" ]; then
    echo "Error: sample_test.csv not found. Please run from project root."
    exit 1
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p images/sample
mkdir -p embeddings/sample
mkdir -p models
mkdir -p submissions

# Step 1: Download sample images
echo "Step 1: Downloading sample images..."
python src/download_images.py \
    --input student_resource/dataset/sample_test.csv \
    --out_dir images/sample \
    --id_col sample_id \
    --url_col image_link \
    --workers 4

echo

# Step 2: Extract optimized embeddings
echo "Step 2: Extracting optimized embeddings..."
python src/extract_embeddings.py \
    --csv student_resource/dataset/sample_test.csv \
    --image_dir images/sample \
    --out_dir embeddings/sample

echo

# Step 3: Train optimized model (quick mode)
echo "Step 3: Training optimized model (quick mode)..."
python src/train_baseline.py \
    --train_csv student_resource/dataset/small_train.csv \
    --emb_dir embeddings/small_train \
    --out_dir models \
    --quick 1 \
    --cv_folds 3 \
    --model optimized

echo

# Step 4: Generate predictions
echo "Step 4: Generating predictions..."
python src/predict.py \
    --model models/baseline_lgb.pkl \
    --test_csv student_resource/dataset/sample_test.csv \
    --emb_dir embeddings/sample \
    --out submissions/sample_out.csv

echo

# Step 5: Validate output
echo "Step 5: Validating output..."
if [ -f "submissions/sample_out.csv" ]; then
    echo "✓ Output file created successfully"
    echo "Sample predictions:"
    head -5 submissions/sample_out.csv
    echo
    echo "Total predictions: $(wc -l < submissions/sample_out.csv)"
else
    echo "✗ Output file not created"
    exit 1
fi

echo
echo "=== Sample Pipeline Test Completed Successfully! ==="
echo "Next steps:"
echo "1. Review the sample predictions in submissions/sample_out.csv"
echo "2. Compare with expected format in student_resource/dataset/sample_test_out.csv"
echo "3. Run full pipeline on complete dataset when ready"
echo "4. Submit your solution!"
