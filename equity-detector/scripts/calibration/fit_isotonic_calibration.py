"""
Fit and save isotonic calibration model for production use
This script loads validation predictions and fits an isotonic regression model
"""

import numpy as np
import pickle
from sklearn.isotonic import IsotonicRegression
from pathlib import Path

# Paths
EVAL_DIR = Path(__file__).parent
VAL_PROBS_PATH = EVAL_DIR / "val_probs.npy"
VAL_LABELS_PATH = EVAL_DIR / "val_labels.npy"
OUTPUT_PATH = EVAL_DIR / "isotonic_calibrator.pkl"

def main():
    """Fit and save isotonic calibration model"""
    
    print("=" * 80)
    print("FITTING ISOTONIC CALIBRATION MODEL")
    print("=" * 80)
    
    # Load validation data
    print(f"\nLoading validation probabilities from: {VAL_PROBS_PATH}")
    val_probs = np.load(VAL_PROBS_PATH)
    print(f"  Shape: {val_probs.shape}")
    print(f"  Range: [{val_probs.min():.4f}, {val_probs.max():.4f}]")
    
    print(f"\nLoading validation labels from: {VAL_LABELS_PATH}")
    val_labels = np.load(VAL_LABELS_PATH)
    print(f"  Shape: {val_labels.shape}")
    print(f"  Positive samples: {val_labels.sum()} ({val_labels.sum()/len(val_labels)*100:.2f}%)")
    
    # We're using binary classification with "any_toxic" label
    # For multi-label, we need to aggregate to binary
    if len(val_labels.shape) > 1 and val_labels.shape[1] > 1:
        # Aggregate: any toxic label = 1
        val_labels_binary = (val_labels.sum(axis=1) > 0).astype(int)
        print(f"\nAggregated to binary 'any_toxic' label")
        print(f"  Positive samples: {val_labels_binary.sum()} ({val_labels_binary.sum()/len(val_labels_binary)*100:.2f}%)")
    else:
        val_labels_binary = val_labels.ravel()
    
    # For probabilities, we also need to aggregate if multi-label
    if len(val_probs.shape) > 1 and val_probs.shape[1] > 1:
        # Aggregate: max probability across all toxic labels
        val_probs_binary = val_probs.max(axis=1)
        print(f"\nAggregated probabilities (taking max across labels)")
        print(f"  Range: [{val_probs_binary.min():.4f}, {val_probs_binary.max():.4f}]")
    else:
        val_probs_binary = val_probs.ravel()
    
    # Fit isotonic regression
    print("\n" + "=" * 80)
    print("FITTING ISOTONIC REGRESSION")
    print("=" * 80)
    
    isotonic_model = IsotonicRegression(out_of_bounds='clip')
    isotonic_model.fit(val_probs_binary, val_labels_binary)
    
    print(f"✅ Isotonic regression fitted successfully")
    print(f"   Number of calibration points: {len(isotonic_model.X_thresholds_)}")
    print(f"   Input range: [{isotonic_model.X_min_:.4f}, {isotonic_model.X_max_:.4f}]")
    print(f"   Output range: [{isotonic_model.y_min_:.4f}, {isotonic_model.y_max_:.4f}]")
    
    # Test calibration on validation set
    print("\n" + "=" * 80)
    print("VALIDATION PERFORMANCE")
    print("=" * 80)
    
    val_probs_calibrated = isotonic_model.transform(val_probs_binary)
    
    print(f"\nUncalibrated probabilities:")
    print(f"  Mean: {val_probs_binary.mean():.4f}")
    print(f"  Std:  {val_probs_binary.std():.4f}")
    print(f"  Range: [{val_probs_binary.min():.4f}, {val_probs_binary.max():.4f}]")
    
    print(f"\nCalibrated probabilities:")
    print(f"  Mean: {val_probs_calibrated.mean():.4f}")
    print(f"  Std:  {val_probs_calibrated.std():.4f}")
    print(f"  Range: [{val_probs_calibrated.min():.4f}, {val_probs_calibrated.max():.4f}]")
    
    # Apply optimal threshold (0.40 for isotonic)
    optimal_threshold = 0.40
    predictions = (val_probs_calibrated >= optimal_threshold).astype(int)
    
    # Compute metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(val_labels_binary, predictions)
    precision = precision_score(val_labels_binary, predictions)
    recall = recall_score(val_labels_binary, predictions)
    f1 = f1_score(val_labels_binary, predictions)
    
    print(f"\nWith threshold = {optimal_threshold}:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
    
    # Save model
    print("\n" + "=" * 80)
    print("SAVING MODEL")
    print("=" * 80)
    
    calibration_data = {
        'model': isotonic_model,
        'optimal_threshold': optimal_threshold,
        'validation_accuracy': accuracy,
        'validation_f1': f1,
        'validation_precision': precision,
        'validation_recall': recall,
        'method': 'isotonic_regression',
        'description': 'Isotonic regression calibration for binary any_toxic classification'
    }
    
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(calibration_data, f)
    
    print(f"✅ Calibration model saved to: {OUTPUT_PATH}")
    print(f"   File size: {OUTPUT_PATH.stat().st_size:,} bytes")
    
    print("\n" + "=" * 80)
    print("CALIBRATION MODEL READY FOR PRODUCTION")
    print("=" * 80)
    print(f"\nTo use this model:")
    print(f"1. Load: calibration_data = pickle.load(open('{OUTPUT_PATH}', 'rb'))")
    print(f"2. Calibrate: probs_cal = calibration_data['model'].transform(probs)")
    print(f"3. Classify: predictions = (probs_cal >= {optimal_threshold})")
    
    return calibration_data

if __name__ == "__main__":
    main()

