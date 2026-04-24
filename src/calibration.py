import numpy as np
import pandas as pd
from scipy import interpolate

class PredictionCalibrator:
    """
    Calibrates predictions to match training distribution.
    Reduces extreme errors and improves SMAPE.
    """
    
    def __init__(self, train_prices):
        """
        Args:
            train_prices: array of training set prices
        """
        self.train_prices = np.array(train_prices)
        
        # Compute clipping bounds (1st and 99th percentiles)
        self.clip_min = np.percentile(self.train_prices, 1)
        self.clip_max = np.percentile(self.train_prices, 99)
        
        # Compute quantile mapping
        self.quantiles = np.linspace(0, 1, 100)
        self.train_quantiles = np.percentile(self.train_prices, self.quantiles * 100)
        
        print(f"Calibrator initialized:")
        print(f"  Clip range: ${self.clip_min:.2f} - ${self.clip_max:.2f}")
        print(f"  Train mean: ${self.train_prices.mean():.2f}")
        print(f"  Train median: ${np.median(self.train_prices):.2f}")
    
    def clip_predictions(self, predictions):
        """
        Simple clipping to train percentiles.
        """
        return np.clip(predictions, self.clip_min, self.clip_max)
    
    def quantile_calibrate(self, predictions):
        """
        Map predictions to match training distribution quantiles.
        """
        # Compute quantiles of predictions
        pred_sorted = np.sort(predictions)
        pred_quantiles = np.percentile(predictions, self.quantiles * 100)
        
        # Create interpolation function from pred quantiles to train quantiles
        # This maps prediction distribution to training distribution
        calibrator = interpolate.interp1d(
            pred_quantiles, 
            self.train_quantiles,
            kind='linear',
            bounds_error=False,
            fill_value=(self.train_quantiles[0], self.train_quantiles[-1])
        )
        
        calibrated = calibrator(predictions)
        return calibrated
    
    def isotonic_calibrate(self, predictions, validation_prices):
        """
        Isotonic regression calibration (requires validation set).
        """
        from sklearn.isotonic import IsotonicRegression
        
        iso_reg = IsotonicRegression(out_of_bounds='clip')
        iso_reg.fit(predictions, validation_prices)
        
        self.isotonic_model = iso_reg
        return iso_reg.predict(predictions)
    
    def calibrate_conservative(self, predictions, alpha=0.8):
        """
        Conservative calibration: blend clipped + quantile-mapped predictions.
        
        Args:
            predictions: raw predictions
            alpha: weight for clipped predictions (1-alpha for quantile-mapped)
        """
        clipped = self.clip_predictions(predictions)
        quantile_mapped = self.quantile_calibrate(predictions)
        
        # Blend
        calibrated = alpha * clipped + (1 - alpha) * quantile_mapped
        
        return calibrated
    
    def calibrate_full(self, predictions):
        """
        Full calibration pipeline: clip then quantile map.
        """
        clipped = self.clip_predictions(predictions)
        calibrated = self.quantile_calibrate(clipped)
        return calibrated


def apply_calibration_pipeline(predictions_file, train_csv_path, output_file, method='conservative'):
    """
    Apply calibration to a predictions file.
    
    Args:
        predictions_file: path to CSV with sample_id, price columns
        train_csv_path: path to train.csv
        output_file: where to save calibrated predictions
        method: 'clip', 'quantile', 'conservative', or 'full'
    """
    print("="*70)
    print(f"CALIBRATING PREDICTIONS - Method: {method}")
    print("="*70)
    
    # Load data
    print("\n1. Loading data...")
    train = pd.read_csv(train_csv_path)
    preds = pd.read_csv(predictions_file)
    
    print(f"   Train samples: {len(train)}")
    print(f"   Prediction samples: {len(preds)}")
    
    # Initialize calibrator
    print("\n2. Initializing calibrator...")
    calibrator = PredictionCalibrator(train['price'].values)
    
    # Apply calibration
    print(f"\n3. Applying {method} calibration...")
    original_prices = preds['price'].values
    
    if method == 'clip':
        calibrated_prices = calibrator.clip_predictions(original_prices)
    elif method == 'quantile':
        calibrated_prices = calibrator.quantile_calibrate(original_prices)
    elif method == 'conservative':
        calibrated_prices = calibrator.calibrate_conservative(original_prices, alpha=0.7)
    elif method == 'full':
        calibrated_prices = calibrator.calibrate_full(original_prices)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Statistics
    print(f"\n4. Calibration statistics:")
    print(f"   Original:   mean=${original_prices.mean():.2f}, std=${original_prices.std():.2f}")
    print(f"   Calibrated: mean=${calibrated_prices.mean():.2f}, std=${calibrated_prices.std():.2f}")
    print(f"   Train:      mean=${train['price'].mean():.2f}, std=${train['price'].std():.2f}")
    
    # Save
    result = pd.DataFrame({
        'sample_id': preds['sample_id'],
        'price': calibrated_prices
    })
    result.to_csv(output_file, index=False)
    
    print(f"\n5. Saved to: {output_file}")
    print("="*70)
    
    return calibrated_prices


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python calibration.py <predictions_file> <output_file> [method]")
        print("Methods: clip, quantile, conservative (default), full")
        sys.exit(1)
    
    predictions_file = sys.argv[1]
    output_file = sys.argv[2]
    method = sys.argv[3] if len(sys.argv) > 3 else 'conservative'
    
    apply_calibration_pipeline(
        predictions_file,
        "student_resource/dataset/train.csv",
        output_file,
        method=method
    )
