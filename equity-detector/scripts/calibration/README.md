# Calibration Scripts

This directory contains scripts for model calibration and probability calibration testing.

---

## Overview

These scripts are used to calibrate the BERT model's output probabilities to improve prediction reliability. Calibration ensures that predicted probabilities match actual frequencies (e.g., when the model says 70% confidence, it should be correct ~70% of the time).

---

## Scripts

### `fit_isotonic_calibration.py`
**Purpose:** Fits an isotonic regression calibration model to the validation set predictions.

**What it does:**
- Loads validation set logits and labels
- Applies isotonic regression calibration
- Saves the calibration model to `evaluation/isotonic_calibration.json`
- Generates calibration curve visualizations

**Usage:**
```bash
python scripts/calibration/fit_isotonic_calibration.py
```

**Outputs:**
- `evaluation/isotonic_calibration.json` - Calibration model parameters
- Calibration reliability plots in `visualizations/calibrated/`

---

### `fit_isotonic_simple.py`
**Purpose:** Simplified version of isotonic calibration fitting.

**What it does:**
- Streamlined calibration process
- Fewer dependencies and simpler output
- Good for quick calibration experiments

**Usage:**
```bash
python scripts/calibration/fit_isotonic_simple.py
```

**When to use:**
- Quick prototyping
- Testing calibration concepts
- Minimal output needs

---

### `test_calibration.py`
**Purpose:** Tests and evaluates different calibration methods.

**What it does:**
- Compares uncalibrated vs. calibrated predictions
- Tests temperature scaling, Platt scaling, and isotonic regression
- Generates comprehensive metrics and visualizations
- Outputs reliability diagrams showing calibration improvement

**Usage:**
```bash
python scripts/calibration/test_calibration.py
```

**Outputs:**
- Calibration metrics for each method
- Reliability plots showing before/after calibration
- Comparison charts in `visualizations/`

---

## Calibration Methods

### Isotonic Regression (Recommended)
- **Best for:** Non-parametric calibration
- **Pros:** Flexible, handles complex calibration curves
- **Cons:** Requires sufficient validation data
- **Result:** Validation accuracy improved from 89.94% → 96.84%

### Temperature Scaling
- **Best for:** Simple, parametric calibration
- **Pros:** Fast, single parameter
- **Cons:** Assumes uniform miscalibration
- **Use case:** Quick calibration baseline

### Platt Scaling
- **Best for:** Logistic calibration
- **Pros:** Works well for SVMs and similar models
- **Cons:** May overfit on small datasets
- **Use case:** Alternative to temperature scaling

---

## Workflow

### Initial Calibration (One-time)
1. Train BERT model on training set
2. Generate predictions on validation set
3. Run `fit_isotonic_calibration.py` to create calibration model
4. Save calibration model for production use

### Testing Calibration
1. Run `test_calibration.py` to compare methods
2. Review reliability diagrams
3. Select best method based on metrics
4. Update production config

### Production Deployment
- Calibration parameters are stored in `evaluation/calibration_params.json`
- Isotonic model is stored in `evaluation/isotonic_calibration.json`
- Apply calibration in production inference pipeline

---

## Key Metrics

### Expected Calibration Error (ECE)
- Measures calibration quality
- Lower is better
- Target: < 0.05

### Reliability Diagram
- Visual check of calibration
- Diagonal line = perfect calibration
- Shows if model is over/under-confident

### Brier Score
- Measures prediction accuracy
- Lower is better
- Combines calibration + discrimination

---

## Dependencies

All scripts require:
- `numpy`
- `scikit-learn`
- `matplotlib`
- `pandas`

See `requirements.txt` in project root.

---

## Related Documentation

- **Calibration Analysis**: `docs/calibration.md`
- **Model Card**: `docs/model_card.md`
- **Evaluation Results**: `evaluation/metrics_summary.csv`
- **Production Config**: `evaluation/calibration_params.json`

---

## Notes

- Calibration should be performed on a held-out validation set, **never** on the test set
- The validation set used for calibration should be different from training data
- Recalibration may be needed if the model is retrained or data distribution changes
- Always verify calibration on a separate test set after fitting

---

**Last Updated:** October 24, 2025

