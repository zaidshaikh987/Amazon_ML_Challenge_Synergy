@echo off
REM Retrain with fixed images and improved model

echo ============================================
echo Retraining with 100%% image coverage
echo ============================================

echo Step 1: Re-extracting embeddings for training data...
python src\extract_embeddings.py --csv student_resource\dataset\small_train.csv --image_dir images\small_train --out_dir embeddings\small_train
if errorlevel 1 (
    echo Error in embedding extraction
    exit /b 1
)

echo.
echo Step 2: Re-extracting embeddings for test data...
python src\extract_embeddings.py --csv student_resource\dataset\sample_test.csv --image_dir images\sample --out_dir embeddings\sample
if errorlevel 1 (
    echo Error in embedding extraction
    exit /b 1
)

echo.
echo Step 3: Training improved model...
python src\train_baseline.py --train_csv student_resource\dataset\small_train.csv --emb_dir embeddings\small_train --out_dir models --quick 0 --cv_folds 5 --model optimized
if errorlevel 1 (
    echo Error in training
    exit /b 1
)

echo.
echo Step 4: Generating predictions...
python src\predict.py --model models\baseline_lgb.pkl --test_csv student_resource\dataset\sample_test.csv --emb_dir embeddings\sample --out submissions\sample_out.csv --id_col sample_id
if errorlevel 1 (
    echo Error in prediction
    exit /b 1
)

echo.
echo ============================================
echo Retraining Complete!
echo ============================================
pause
