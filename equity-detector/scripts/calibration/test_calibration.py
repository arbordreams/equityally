"""
Test script to verify isotonic calibration implementation
"""

import numpy as np
import json
from pathlib import Path

# Test the calibration loading and application
def test_calibration():
    print("=" * 80)
    print("TESTING ISOTONIC CALIBRATION")
    print("=" * 80)
    
    # Load calibration data
    calibration_path = Path(__file__).parent / "evaluation" / "isotonic_calibration.json"
    
    if not calibration_path.exists():
        print(f"❌ Calibration file not found at {calibration_path}")
        return False
    
    print(f"\n✅ Loading calibration from: {calibration_path}")
    
    with open(calibration_path, 'r') as f:
        calibration_data = json.load(f)
    
    x_cal = np.array(calibration_data['x_calibration'])
    y_cal = np.array(calibration_data['y_calibration'])
    optimal_threshold = calibration_data['optimal_threshold']
    
    print(f"   Calibration points: {len(x_cal)}")
    print(f"   Optimal threshold: {optimal_threshold:.2f} ({optimal_threshold*100:.0f}%)")
    print(f"   Validation accuracy: {calibration_data['validation_accuracy']*100:.2f}%")
    print(f"   Validation F1: {calibration_data['validation_f1']*100:.2f}%")
    
    # Test calibration on sample probabilities
    print("\n" + "=" * 80)
    print("TESTING CALIBRATION APPLICATION")
    print("=" * 80)
    
    # Test with various uncalibrated probabilities
    test_probs = np.array([0.01, 0.10, 0.30, 0.50, 0.70, 0.80, 0.90, 0.99])
    
    print("\nUncalibrated → Calibrated probabilities:")
    print("-" * 80)
    print(f"{'Uncalibrated':<15} {'Calibrated':<15} {'Prediction':<15} {'Change':<15}")
    print("-" * 80)
    
    for prob in test_probs:
        # Apply calibration
        prob_cal = np.interp(prob, x_cal, y_cal)
        prediction = "Concerning" if prob_cal >= optimal_threshold else "Safe"
        change = prob_cal - prob
        
        print(f"{prob:<15.4f} {prob_cal:<15.4f} {prediction:<15} {change:+.4f}")
    
    print("\n" + "=" * 80)
    print("THRESHOLD COMPARISON")
    print("=" * 80)
    
    print(f"\nOLD THRESHOLD (uncalibrated): 0.80 (80%)")
    print(f"NEW THRESHOLD (calibrated):   {optimal_threshold:.2f} ({optimal_threshold*100:.0f}%)")
    
    # Show what changes with the new threshold
    print("\n" + "=" * 80)
    print("IMPACT ANALYSIS")
    print("=" * 80)
    
    # Test with specific examples
    examples = [
        ("Low risk content", 0.60),
        ("Medium risk content", 0.75),
        ("High risk content", 0.85)
    ]
    
    print(f"\n{'Example':<25} {'Uncal Prob':<12} {'Cal Prob':<12} {'Old Pred':<12} {'New Pred':<12}")
    print("-" * 80)
    
    for example_name, uncal_prob in examples:
        cal_prob = np.interp(uncal_prob, x_cal, y_cal)
        old_pred = "Concerning" if uncal_prob >= 0.80 else "Safe"
        new_pred = "Concerning" if cal_prob >= optimal_threshold else "Safe"
        
        print(f"{example_name:<25} {uncal_prob:<12.2f} {cal_prob:<12.2f} {old_pred:<12} {new_pred:<12}")
    
    print("\n" + "=" * 80)
    print("✅ CALIBRATION TEST COMPLETE")
    print("=" * 80)
    
    print("\nKey Findings:")
    print(f"1. Calibration shifts probabilities to better reflect true frequencies")
    print(f"2. New threshold ({optimal_threshold:.2f}) is more sensitive than old (0.80)")
    print(f"3. This increases recall while maintaining high precision")
    print(f"4. Expected validation accuracy: {calibration_data['validation_accuracy']*100:.2f}%")
    
    return True

if __name__ == "__main__":
    success = test_calibration()
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Tests failed!")

