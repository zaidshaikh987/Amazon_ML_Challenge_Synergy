@echo off
REM Production pipeline for 75K dataset or sample dataset
REM Usage: run_sample.bat [sample|full]

set MODE=%1
if "%MODE%"=="" set MODE=sample

if "%MODE%"=="full" (
    echo === Amazon ML Challenge 2025 - FULL 75K PRODUCTION PIPELINE ===
    echo WARNING: This will take 2-4 hours and download ~10GB of data
    echo Press Ctrl+C to cancel, or
    pause
    set TRAIN_CSV=student_resource\dataset\train.csv
    set TEST_CSV=student_resource\dataset\test.csv
    set TRAIN_DIR=train
    set TEST_DIR=test
    set ID_COL=sample_id
    set CV_FOLDS=5
    set QUICK=0
    set WORKERS=100
) else (
    echo === Amazon ML Challenge 2025 - Sample Pipeline Test ===
    set TRAIN_CSV=student_resource\dataset\small_train.csv
    set TEST_CSV=student_resource\dataset\sample_test.csv
    set TRAIN_DIR=small_train
    set TEST_DIR=sample
    set ID_COL=sample_id
    set CV_FOLDS=3
    set QUICK=1
    set WORKERS=100
)

echo Starting pipeline in %MODE% mode...
echo.

REM Check if we're in the right directory
if not exist "student_resource\dataset\sample_test.csv" (
    echo Error: sample_test.csv not found. Please run from project root.
    exit /b 1
)

REM Create necessary directories
echo Creating directories...
if not exist "images\%TRAIN_DIR%" mkdir "images\%TRAIN_DIR%"
if not exist "images\%TEST_DIR%" mkdir "images\%TEST_DIR%"
if not exist "embeddings\%TRAIN_DIR%" mkdir "embeddings\%TRAIN_DIR%"
if not exist "embeddings\%TEST_DIR%" mkdir "embeddings\%TEST_DIR%"
if not exist "models" mkdir "models"
if not exist "submissions" mkdir "submissions"

REM Step 1: Download training images
echo Step 1: Downloading training images...
python src\download_images.py --input %TRAIN_CSV% --out_dir images\%TRAIN_DIR% --id_col %ID_COL% --url_col image_link --workers %WORKERS% --retries 5
if errorlevel 1 (
    echo Error in training image download step
    exit /b 1
)

echo.

REM Step 2: Extract embeddings for training data
echo Step 2: Extracting training embeddings...
python src\extract_embeddings.py --csv %TRAIN_CSV% --image_dir images\%TRAIN_DIR% --out_dir embeddings\%TRAIN_DIR%
if errorlevel 1 (
    echo Error in training embedding extraction step
    exit /b 1
)

echo.

REM Step 3: Train model
echo Step 3: Training LightGBM model with optimized hyperparameters...
python src\train_baseline.py --train_csv %TRAIN_CSV% --emb_dir embeddings\%TRAIN_DIR% --out_dir models --quick %QUICK% --cv_folds %CV_FOLDS% --model optimized
if errorlevel 1 (
    echo Error in model training step
    exit /b 1
)

echo.

REM Step 4: Download test images
echo Step 4: Downloading test images...
python src\download_images.py --input %TEST_CSV% --out_dir images\%TEST_DIR% --id_col %ID_COL% --url_col image_link --workers %WORKERS% --retries 5
if errorlevel 1 (
    echo Error in test image download step
    exit /b 1
)

echo.

REM Step 5: Extract embeddings for test data
echo Step 5: Extracting test embeddings...
python src\extract_embeddings.py --csv %TEST_CSV% --image_dir images\%TEST_DIR% --out_dir embeddings\%TEST_DIR%
if errorlevel 1 (
    echo Error in test embedding extraction step
    exit /b 1
)

echo.

REM Step 6: Generate predictions
echo Step 6: Generating predictions...
python src\predict.py --model models\baseline_lgb.pkl --test_csv %TEST_CSV% --emb_dir embeddings\%TEST_DIR% --out submissions\%MODE%_predictions.csv --id_col %ID_COL%
if errorlevel 1 (
    echo Error in prediction step
    exit /b 1
)

echo.

REM Step 7: Validate output
echo Step 7: Validating output...
if exist "submissions\%MODE%_predictions.csv" (
    echo ✓ Output file created successfully: submissions\%MODE%_predictions.csv
    echo.
    echo First 10 predictions:
    powershell -Command "Get-Content submissions\%MODE%_predictions.csv | Select-Object -First 11"
    echo.
    echo Total predictions: 
    find /c /v "" "submissions\%MODE%_predictions.csv"
    echo.
) else (
    echo ✗ Output file not created
    exit /b 1
)

echo.
echo ============================================================
echo Pipeline Completed Successfully in %MODE% mode!
echo ============================================================
if "%MODE%"=="full" (
    echo Output: submissions\full_predictions.csv
    echo Ready for submission to Amazon ML Challenge!
) else (
    echo Output: submissions\sample_predictions.csv
    echo To run on full 75K dataset: run_sample.bat full
)
echo ============================================================
pause
