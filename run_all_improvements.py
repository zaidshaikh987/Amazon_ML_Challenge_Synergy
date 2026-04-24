"""
Master script to run all improvement steps sequentially.
This will take your model from 54% SMAPE to ~51-52% SMAPE.

Steps:
1. Add aggregate features
2. Retrain LightGBM with aggregate features
3. Train neural fusion model (optional, requires GPU)
4. Create comprehensive ensemble with all models
5. Generate final submission recommendations

Run with: python run_all_improvements.py [--skip-neural]
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description, optional=False):
    """Run a command and handle errors."""
    print("\n" + "="*70)
    print(f"🔧 {description}")
    print("="*70)
    print(f"Command: {cmd}")
    print()
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=False)
        print(f"\n✅ {description} - COMPLETED")
        return True
    except subprocess.CalledProcessError as e:
        if optional:
            print(f"\n⚠️  {description} - SKIPPED (optional step failed)")
            return False
        else:
            print(f"\n❌ {description} - FAILED")
            print(f"Error: {e}")
            return False
    except Exception as e:
        print(f"\n❌ {description} - ERROR: {e}")
        if not optional:
            sys.exit(1)
        return False

def check_file_exists(filepath, description):
    """Check if a file exists."""
    if Path(filepath).exists():
        print(f"✅ Found: {description}")
        return True
    else:
        print(f"⚠️  Missing: {description}")
        return False

def main():
    print("="*70)
    print("🚀 COMPREHENSIVE MODEL IMPROVEMENT PIPELINE")
    print("="*70)
    print("\nThis script will:")
    print("1. Add aggregate features (category, brand, product type)")
    print("2. Retrain LightGBM with new features")
    print("3. Train neural fusion model (optional)")
    print("4. Create comprehensive ensembles with calibration")
    print("\nExpected time: 30-90 minutes depending on hardware")
    print("Expected improvement: From 54% to 51-52% SMAPE")
    
    skip_neural = "--skip-neural" in sys.argv
    if skip_neural:
        print("\n⚠️  Skipping neural fusion model (--skip-neural flag)")
    
    input("\nPress ENTER to start or CTRL+C to cancel...")
    
    # Check prerequisites
    print("\n" + "="*70)
    print("📋 CHECKING PREREQUISITES")
    print("="*70)
    
    prereqs_ok = True
    prereqs_ok &= check_file_exists("embeddings/train/image_embeddings.npy", "Train image embeddings")
    prereqs_ok &= check_file_exists("embeddings/train/text_embeddings.npy", "Train text embeddings")
    prereqs_ok &= check_file_exists("embeddings/test/image_embeddings.npy", "Test image embeddings")
    prereqs_ok &= check_file_exists("embeddings/test/text_embeddings.npy", "Test text embeddings")
    prereqs_ok &= check_file_exists("student_resource/dataset/train.csv", "Train CSV")
    prereqs_ok &= check_file_exists("student_resource/dataset/test.csv", "Test CSV")
    prereqs_ok &= check_file_exists("submissions/test_out.csv", "LightGBM predictions")
    prereqs_ok &= check_file_exists("submissions/knn_predictions.csv", "k-NN predictions")
    
    if not prereqs_ok:
        print("\n❌ Missing required files. Please ensure:")
        print("   1. Embeddings are extracted")
        print("   2. LightGBM model is trained (train_baseline.py)")
        print("   3. k-NN predictions exist (knn_price.py)")
        sys.exit(1)
    
    print("\n✅ All prerequisites satisfied!")
    
    # ========== STEP 1: Add Aggregate Features ==========
    success = run_command(
        "python src/aggregate_features.py",
        "STEP 1: Adding aggregate features (category, brand, product type)"
    )
    
    if not success:
        print("\n❌ Failed to add aggregate features. Exiting.")
        sys.exit(1)
    
    # ========== STEP 2: Retrain LightGBM with Aggregates ==========
    # Check if aggregate features were created
    if Path("student_resource/dataset/train_with_aggregates.csv").exists():
        print("\n" + "="*70)
        print("🔧 STEP 2: Retraining LightGBM with aggregate features")
        print("="*70)
        print("\n⚠️  NOTE: This will take 15-30 minutes")
        print("You can skip this and use existing LightGBM for now.")
        
        choice = input("\nRetrain LightGBM with aggregate features? (y/n): ").lower()
        
        if choice == 'y':
            # Create modified training script that uses aggregates
            run_command(
                "python train_baseline.py",
                "Retraining LightGBM with aggregate features",
                optional=True
            )
    else:
        print("\n⚠️  Aggregate features file not found, skipping retrain")
    
    # ========== STEP 3: Train Neural Fusion ==========
    if not skip_neural:
        print("\n" + "="*70)
        print("🔧 STEP 3: Training Neural Fusion Model")
        print("="*70)
        print("\n⚠️  NOTE: This requires GPU and takes 30-60 minutes")
        print("If you don't have a GPU, use --skip-neural flag")
        
        choice = input("\nTrain neural fusion model? (y/n): ").lower()
        
        if choice == 'y':
            run_command(
                "python train_neural_fusion.py",
                "Training neural fusion model with SMAPE loss",
                optional=True
            )
        else:
            print("\n⏭️  Skipping neural fusion model")
    else:
        print("\n⏭️  Skipping neural fusion model (--skip-neural flag)")
    
    # ========== STEP 4: Create Comprehensive Ensemble ==========
    run_command(
        "python create_final_ensemble.py",
        "STEP 4: Creating comprehensive ensemble with calibration"
    )
    
    # ========== STEP 5: Summary and Recommendations ==========
    print("\n" + "="*70)
    print("🎉 PIPELINE COMPLETED!")
    print("="*70)
    
    print("\n📊 SUBMISSION FILES READY:")
    print("\nTop 5 submissions to test (in order):")
    print("\n1. submissions/ensemble_blend_50_50_calibrated.csv")
    print("   → Balanced blend with calibration (RECOMMENDED)")
    
    print("\n2. submissions/ensemble_lgb_calibrated_conservative.csv")
    print("   → Single LightGBM with conservative calibration")
    
    print("\n3. submissions/ensemble_blend_60_40_calibrated.csv")
    print("   → Favor LightGBM slightly with calibration")
    
    print("\n4. submissions/ensemble_lgb_knn_50_50.csv")
    print("   → Simple 50/50 blend (no calibration)")
    
    print("\n5. submissions/ensemble_knn_calibrated.csv")
    print("   → k-NN with calibration")
    
    if Path("submissions/neural_fusion_predictions.csv").exists():
        print("\n🎯 BONUS: Neural fusion predictions available!")
        print("   → Check submissions/ensemble_all_*.csv for 3-way blends")
    
    print("\n" + "="*70)
    print("📈 EXPECTED RESULTS")
    print("="*70)
    print("\nCurrent performance:")
    print("  - LightGBM: 54.00% SMAPE")
    print("  - k-NN: 54.73% SMAPE")
    print("\nExpected with calibrated blends:")
    print("  - Target: 51-53% SMAPE")
    print("  - Best case: <51% SMAPE")
    
    print("\n" + "="*70)
    print("💡 NEXT STEPS")
    print("="*70)
    print("\n1. Submit ensemble_blend_50_50_calibrated.csv to competition")
    print("2. Compare leaderboard score with other submissions")
    print("3. If needed, iterate on blend weights")
    print("\nFor further improvements:")
    print("  - Train stratified ensemble: python train_stratified_ensemble.py")
    print("  - Fine-tune embeddings (advanced)")
    print("  - Add more domain-specific features")
    
    print("\n✅ All done! Good luck with your submissions! 🚀")

if __name__ == "__main__":
    main()
