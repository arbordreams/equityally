"""
Fit and save isotonic calibration using only numpy
Creates a lookup table for calibration without requiring sklearn at runtime
"""

import numpy as np
import json
from pathlib import Path

# Paths
EVAL_DIR = Path(__file__).parent
VAL_PROBS_PATH = EVAL_DIR / "val_probs.npy"
VAL_LABELS_PATH = EVAL_DIR / "val_labels.npy"
OUTPUT_PATH = EVAL_DIR / "isotonic_calibration.json"

def fit_isotonic_simple(y_true, y_pred):
    """
    Simple isotonic regression implementation
    Returns calibration mapping points (x, y) for interpolation
    """
    # Sort by predicted probabilities
    order = np.argsort(y_pred)
    y_pred_sorted = y_pred[order]
    y_true_sorted = y_true[order]
    
    # Pool adjacent violators algorithm (simple version)
    n = len(y_pred_sorted)
    
    # Initialize
    y_cal = np.zeros(n)
    
    # Use pooling to ensure monotonicity
    # Simple approach: compute cumulative means
    for i in range(n):
        # Average over window to smooth and ensure isotonicity
        window_size = min(100, n - i)  # Adaptive window
        y_cal[i] = np.mean(y_true_sorted[i:i+window_size])
    
    # Ensure strict monotonicity by taking cumulative maximum
    for i in range(1, n):
        if y_cal[i] < y_cal[i-1]:
            y_cal[i] = y_cal[i-1]
    
    # Sample calibration points (use ~1000 points to reduce size)
    num_points = min(1000, n)
    indices = np.linspace(0, n-1, num_points, dtype=int)
    
    x_cal = y_pred_sorted[indices]
    y_cal_sampled = y_cal[indices]
    
    return x_cal, y_cal_sampled

def main():
    """Fit and save isotonic calibration"""
    
    print("=" * 80)
    print("FITTING ISOTONIC CALIBRATION (NUMPY-ONLY VERSION)")
    print("=" * 80)
    
    # Load validation data
    print(f"\nLoading validation probabilities from: {VAL_PROBS_PATH}")
    val_probs = np.load(VAL_PROBS_PATH)
    print(f"  Shape: {val_probs.shape}")
    print(f"  Range: [{val_probs.min():.4f}, {val_probs.max():.4f}]")
    
    print(f"\nLoading validation labels from: {VAL_LABELS_PATH}")
    val_labels = np.load(VAL_LABELS_PATH)
    print(f"  Shape: {val_labels.shape}")
    
    # Aggregate to binary
    if len(val_labels.shape) > 1 and val_labels.shape[1] > 1:
        val_labels_binary = (val_labels.sum(axis=1) > 0).astype(float)
        print(f"\nAggregated to binary 'any_toxic' label")
        print(f"  Positive samples: {val_labels_binary.sum():.0f} ({val_labels_binary.sum()/len(val_labels_binary)*100:.2f}%)")
    else:
        val_labels_binary = val_labels.ravel().astype(float)
    
    if len(val_probs.shape) > 1 and val_probs.shape[1] > 1:
        val_probs_binary = val_probs.max(axis=1)
        print(f"\nAggregated probabilities (taking max across labels)")
        print(f"  Range: [{val_probs_binary.min():.4f}, {val_probs_binary.max():.4f}]")
    else:
        val_probs_binary = val_probs.ravel()
    
    # Fit isotonic calibration
    print("\n" + "=" * 80)
    print("FITTING ISOTONIC REGRESSION")
    print("=" * 80)
    
    x_cal, y_cal = fit_isotonic_simple(val_labels_binary, val_probs_binary)
    
    print(f"✅ Isotonic regression fitted successfully")
    print(f"   Number of calibration points: {len(x_cal)}")
    print(f"   Input range: [{x_cal.min():.4f}, {x_cal.max():.4f}]")
    print(f"   Output range: [{y_cal.min():.4f}, {y_cal.max():.4f}]")
    
    # Apply calibration using interpolation
    print("\n" + "=" * 80)
    print("VALIDATION PERFORMANCE")
    print("=" * 80)
    
    val_probs_calibrated = np.interp(val_probs_binary, x_cal, y_cal)
    
    print(f"\nUncalibrated probabilities:")
    print(f"  Mean: {val_probs_binary.mean():.4f}")
    print(f"  Std:  {val_probs_binary.std():.4f}")
    print(f"  Range: [{val_probs_binary.min():.4f}, {val_probs_binary.max():.4f}]")
    
    print(f"\nCalibrated probabilities:")
    print(f"  Mean: {val_probs_calibrated.mean():.4f}")
    print(f"  Std:  {val_probs_calibrated.std():.4f}")
    print(f"  Range: [{val_probs_calibrated.min():.4f}, {val_probs_calibrated.max():.4f}]")
    
    # Try different thresholds and find optimal
    print("\n" + "=" * 80)
    print("THRESHOLD OPTIMIZATION")
    print("=" * 80)
    
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in np.arange(0.1, 0.9, 0.05):
        predictions = (val_probs_calibrated >= threshold).astype(int)
        
        # Compute F1
        tp = ((predictions == 1) & (val_labels_binary == 1)).sum()
        fp = ((predictions == 1) & (val_labels_binary == 0)).sum()
        fn = ((predictions == 0) & (val_labels_binary == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_precision = precision
            best_recall = recall
    
    print(f"\nOptimal threshold: {best_threshold:.2f}")
    print(f"  F1 Score:  {best_f1:.4f} ({best_f1*100:.2f}%)")
    print(f"  Precision: {best_precision:.4f} ({best_precision*100:.2f}%)")
    print(f"  Recall:    {best_recall:.4f} ({best_recall*100:.2f}%)")
    
    # Final evaluation with optimal threshold
    predictions = (val_probs_calibrated >= best_threshold).astype(int)
    accuracy = (predictions == val_labels_binary).mean()
    
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Save calibration data
    print("\n" + "=" * 80)
    print("SAVING CALIBRATION DATA")
    print("=" * 80)
    
    calibration_data = {
        'x_calibration': x_cal.tolist(),
        'y_calibration': y_cal.tolist(),
        'optimal_threshold': float(best_threshold),
        'validation_accuracy': float(accuracy),
        'validation_f1': float(best_f1),
        'validation_precision': float(best_precision),
        'validation_recall': float(best_recall),
        'method': 'isotonic_regression',
        'description': 'Isotonic regression calibration for binary any_toxic classification',
        'usage': 'Use np.interp(probs, x_calibration, y_calibration) to calibrate'
    }
    
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(calibration_data, f, indent=2)
    
    print(f"✅ Calibration data saved to: {OUTPUT_PATH}")
    print(f"   File size: {OUTPUT_PATH.stat().st_size:,} bytes")
    
    print("\n" + "=" * 80)
    print("CALIBRATION MODEL READY FOR PRODUCTION")
    print("=" * 80)
    print(f"\nTo use this calibration:")
    print(f"1. Load: data = json.load(open('{OUTPUT_PATH}'))")
    print(f"2. Calibrate: probs_cal = np.interp(probs, data['x_calibration'], data['y_calibration'])")
    print(f"3. Classify: predictions = (probs_cal >= {best_threshold:.2f})")
    
    return calibration_data

if __name__ == "__main__":
    main()

