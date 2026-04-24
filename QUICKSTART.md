# 🚀 Quick Start Guide

Get your environment up and running in 5 minutes!

## Step 1: Setup Virtual Environment

### Windows
```batch
setup_venv.bat
```

### Linux/Mac
```bash
chmod +x setup_venv.sh
./setup_venv.sh
```

**Choose CPU or GPU** when prompted.

---

## Step 2: Activate Environment

### Windows
```batch
venv\Scripts\activate.bat
```

### Linux/Mac
```bash
source venv/bin/activate
```

---

## Step 3: Verify Installation

```bash
python verify_setup.py
```

You should see ✅ for all packages.

---

## Step 4: Run Sample Pipeline

### Windows
```batch
run_sample.bat
```

### Linux/Mac
```bash
chmod +x run_sample.sh
./run_sample.sh
```

This will:
1. Download sample images
2. Extract embeddings
3. Train a model
4. Generate predictions

---

## 📁 What You'll Get

After running the sample pipeline:

```
Amazon/
├── images/sample/           # Downloaded images
├── embeddings/sample/       # Text & image embeddings
├── models/                  # Trained LightGBM model
└── submissions/             # Predictions (sample_out.csv)
```

---

## 🎯 Next Steps

### Option A: Full Training Pipeline

```bash
# 1. Download all training images
python src/download_images.py \
    --input student_resource/dataset/train.csv \
    --out_dir images/train \
    --workers 8

# 2. Extract embeddings
python src/extract_embeddings.py \
    --csv student_resource/dataset/train.csv \
    --image_dir images/train \
    --out_dir embeddings/train

# 3. Train model
python src/train_baseline.py \
    --train_csv student_resource/dataset/train.csv \
    --emb_dir embeddings/train \
    --out_dir models \
    --cv_folds 5

# 4. Generate predictions
python src/predict.py \
    --model models/baseline_lgb.pkl \
    --test_csv student_resource/dataset/test.csv \
    --emb_dir embeddings/test \
    --out submissions/test_out.csv
```

### Option B: Quick Experiments

Use `--quick 1` flag for faster training:

```bash
python src/train_baseline.py \
    --train_csv student_resource/dataset/train.csv \
    --emb_dir embeddings/train \
    --out_dir models \
    --quick 1 \
    --cv_folds 3
```

---

## 🔧 Common Commands

### Download Images
```bash
python src/download_images.py \
    --input <csv_file> \
    --out_dir <output_dir> \
    --workers 8
```

### Extract Embeddings
```bash
python src/extract_embeddings.py \
    --csv <csv_file> \
    --image_dir <image_dir> \
    --out_dir <output_dir>
```

### Train Model
```bash
python src/train_baseline.py \
    --train_csv <train_csv> \
    --emb_dir <embeddings_dir> \
    --out_dir <models_dir>
```

### Generate Predictions
```bash
python src/predict.py \
    --model <model_path> \
    --test_csv <test_csv> \
    --emb_dir <embeddings_dir> \
    --out <output_csv>
```

---

## 📊 Expected Performance

**Baseline Model (Sample Data)**
- SMAPE: ~28% 
- Training Time: ~1 minute
- Prediction Time: ~1 second

**Full Model (75k samples)**
- SMAPE: <20% (target)
- Training Time: ~5-10 minutes
- Prediction Time: ~10 seconds

---

## ❓ Troubleshooting

### Environment activation doesn't work
```bash
# Windows
python -m venv venv
venv\Scripts\activate.bat

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Package import fails
```bash
# Verify installation
python verify_setup.py

# Reinstall if needed
pip install -r requirements.txt
```

### CUDA not available (but you have GPU)
```bash
# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Out of memory
```bash
# Use smaller batch size or quick mode
python src/train_baseline.py --quick 1 --cv_folds 3
```

---

## 📚 More Information

- **Full Setup Guide**: See `ENVIRONMENT_SETUP.md`
- **Project Documentation**: See `README.md`
- **Solution Details**: See `SOLUTION_SUMMARY.md`

---

## 💡 Tips

1. **Start Small**: Test with sample data first
2. **Use GPU**: 10x faster with CUDA
3. **Cache Embeddings**: Saves time on re-runs
4. **Monitor Progress**: All scripts show progress bars
5. **Check Logs**: Detailed logging for debugging

---

**Ready to go? Run `setup_venv.bat` and start coding!** 🎉
