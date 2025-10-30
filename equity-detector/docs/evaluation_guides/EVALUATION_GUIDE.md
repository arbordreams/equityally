# BERT Toxicity Model - Evaluation & Calibration Guide

## Overview

This guide provides comprehensive instructions for evaluating and calibrating the pre-trained BERT toxicity classification model. The evaluation includes:

- **Baseline Performance**: Uncalibrated model metrics
- **Post-Hoc Calibration**: Three calibration methods (Temperature, Platt, Isotonic)
- **Threshold Optimization**: Per-label F1 maximization
- **Tokenizer Analysis**: Vocabulary coverage, fragmentation, OOV
- **Extensive Visualization**: 30+ plots and charts
- **Production Recommendations**: Best method, thresholds, deployment config

## Quick Start

### Prerequisites

```bash
# Ensure Python 3.13+ is installed
python3 --version

# Install dependencies
pip install -r requirements.txt
```

### Running the Evaluation

#### Option 1: Complete Pipeline with Logging

```bash
cd /Users/seb/Desktop/EquityLens/equity-detector
bash scripts/run_with_logging.sh
```

This will:
- Run the complete evaluation pipeline
- Save detailed logs to `logs/evaluation_TIMESTAMP.log`
- Generate all visualizations, metrics, and documentation

**Expected Runtime**: 30-60 minutes (depending on CPU/GPU)

#### Option 2: Step-by-Step Execution

```bash
# 1. Check dependencies
PYTHONPATH=venv/lib/python3.13/site-packages python3 scripts/check_and_install_deps.py

# 2. Run main evaluation (Phases 1-6)
PYTHONPATH=venv/lib/python3.13/site-packages python3 scripts/run_evaluation_pipeline.py

# 3. Run tokenizer analysis (Phase 7)
PYTHONPATH=venv/lib/python3.13/site-packages python3 scripts/run_tokenizer_and_summary.py

# 4. Run complete pipeline with documentation (Phases 1-10)
PYTHONPATH=venv/lib/python3.13/site-packages python3 scripts/run_complete_evaluation.py
```

#### Option 3: With asciinema Recording

```bash
bash scripts/run_with_recording.sh
```

This will record the terminal session for documentation and sharing.

## Pipeline Phases

### Phase 1: Data Preparation
- Load validation (159K samples) and test (64K samples) sets
- Compute label prevalence and class imbalance
- Generate visualizations

**Outputs**:
- `visualizations/label_prevalence_val.png`
- `visualizations/label_prevalence_test.png`
- `visualizations/class_imbalance_heatmap.png`
- `docs/dataset_card.md`

### Phase 2: Model Loading
- Load BERT model and tokenizer from `BERT_Model/`
- Verify architecture (6-label multi-label classification)
- Set random seed for reproducibility

### Phase 3: Baseline Inference
- Run uncalibrated inference on validation and test sets
- Save raw logits and probabilities
- Compute baseline metrics (F1, ROC-AUC, PR-AUC, ECE, Brier)

**Outputs**:
- `evaluation/val_logits.npy`, `test_logits.npy`
- `evaluation/val_probs.npy`, `test_probs.npy`
- `evaluation/metrics_val_uncal.json`, `metrics_test_uncal.json`

### Phase 4: Baseline Visualization
- ROC curves (per-label)
- Precision-Recall curves (per-label)
- Reliability diagrams (calibration plots)
- Confusion matrices (aggregated)

**Outputs**:
- `visualizations/val_uncal_roc_perlabel.png`
- `visualizations/test_uncal_roc_perlabel.png`
- `visualizations/val_uncal_pr_perlabel.png`
- `visualizations/test_uncal_pr_perlabel.png`
- `visualizations/val_uncal_reliability.png`
- `visualizations/test_uncal_reliability.png`
- `visualizations/val_uncal_cm_aggregate.png`
- `visualizations/test_uncal_cm_aggregate.png`

### Phase 5: Post-Hoc Calibration

Three calibration methods are fitted on validation and evaluated on test:

#### Temperature Scaling
- Single scalar parameter `T`
- Transforms logits: `probs_cal = sigmoid(logits / T)`
- Fast, simple, effective for well-calibrated models

#### Platt Scaling
- Per-label logistic regression
- Learns `a` and `b`: `P(y=1|logit) = σ(a*logit + b)`
- More flexible than temperature scaling

#### Isotonic Regression
- Per-label non-parametric monotonic mapping
- Most flexible, can overfit on small datasets
- No parametric assumptions

**Outputs** (per method):
- `visualizations/{val|test}_{method}_roc_perlabel.png`
- `visualizations/{val|test}_{method}_pr_perlabel.png`
- `visualizations/{val|test}_{method}_reliability.png`
- `visualizations/{val|test}_{method}_cm_aggregate.png`
- `evaluation/metrics_{method}.json`

**Calibration Parameters**:
- `evaluation/calibration_params.json` - All fitted parameters

### Phase 6: Threshold Optimization

For each label and each calibration method:
- Sweep thresholds from 0.1 to 0.95
- Find threshold maximizing F1 (primary)
- Find threshold minimizing ECE (secondary)
- Find threshold minimizing Brier (secondary)

**Outputs**:
- `visualizations/threshold_sweep_{label}.png` (6 plots)
- `visualizations/threshold_heatmap_all_methods.png`
- `evaluation/thresholds_table.csv`

### Phase 7: Tokenizer Analysis

Comprehensive analysis of BERT tokenizer behavior:
- Token length distributions (validation & test)
- Fragmentation analysis (toxic vs non-toxic)
- Zipf's law verification
- Rare token identification
- OOV/[UNK] frequency

**Outputs**:
- `visualizations/token_length_hist_val.png`
- `visualizations/token_length_hist_test.png`
- `visualizations/token_fragmentation_box_val.png`
- `visualizations/token_fragmentation_box_test.png`
- `visualizations/token_zipf_val.png`
- `visualizations/token_zipf_test.png`
- `visualizations/token_rare_topk_val.png`
- `docs/tokenizer_report.md`

### Phase 8: Summary Dashboards

Comparative visualizations across all methods:
- ROC overlay (uncalibrated vs best calibrated)
- PR overlay
- Reliability diagram comparison
- Metrics bar charts (F1, AUC, ECE, Brier)

**Outputs**:
- `visualizations/summary_roc_overlay.png`
- `visualizations/summary_pr_overlay.png`
- `visualizations/summary_reliability_overlay.png`
- `visualizations/summary_metrics_bars.png`

### Phase 9: Documentation

Comprehensive documentation for all stakeholders:

**For ML Engineers**:
- `docs/calibration.md` - Detailed calibration analysis
- `docs/tokenizer_report.md` - Tokenization insights

**For Data Scientists**:
- `docs/dataset_card.md` - Dataset statistics and schema
- `evaluation/metrics_summary.csv` - All metrics in tabular format

**For Product/Leadership**:
- `docs/onepager.md` - Executive summary with recommendations
- `docs/model_card.md` - Model overview, performance, limitations

**For Developers**:
- `docs/README.md` - Documentation hub with links

### Phase 10: Executive Summary

Console output with:
- Recommended calibration method
- Expected ECE/Brier improvement
- F1/AUC trade-offs
- Per-label optimal thresholds
- Operational considerations
- Tokenizer insights
- Production deployment config

## Outputs Summary

After completion, you will have:

### Visualizations (30+ files)
```
visualizations/
├── label_prevalence_val.png
├── label_prevalence_test.png
├── class_imbalance_heatmap.png
├── val_uncal_roc_perlabel.png
├── test_uncal_roc_perlabel.png
├── val_uncal_pr_perlabel.png
├── test_uncal_pr_perlabel.png
├── val_uncal_reliability.png
├── test_uncal_reliability.png
├── val_uncal_cm_aggregate.png
├── test_uncal_cm_aggregate.png
├── [similar files for temperature, platt, isotonic]
├── threshold_sweep_toxic.png
├── threshold_sweep_severe_toxic.png
├── threshold_sweep_obscene.png
├── threshold_sweep_threat.png
├── threshold_sweep_insult.png
├── threshold_sweep_identity_hate.png
├── threshold_heatmap_all_methods.png
├── token_length_hist_val.png
├── token_length_hist_test.png
├── token_fragmentation_box_val.png
├── token_fragmentation_box_test.png
├── token_zipf_val.png
├── token_zipf_test.png
├── token_rare_topk_val.png
├── summary_roc_overlay.png
├── summary_pr_overlay.png
├── summary_reliability_overlay.png
└── summary_metrics_bars.png
```

### Evaluation Artifacts
```
evaluation/
├── val_logits.npy
├── val_probs.npy
├── val_labels.npy
├── test_logits.npy
├── test_probs.npy
├── test_labels.npy
├── metrics_val_uncal.json
├── metrics_test_uncal.json
├── metrics_temperature.json
├── metrics_platt.json
├── metrics_isotonic.json
├── calibration_params.json
├── thresholds_table.csv
└── metrics_summary.csv
```

### Documentation
```
docs/
├── README.md                 # Documentation hub
├── dataset_card.md          # Dataset overview
├── calibration.md           # Calibration analysis
├── tokenizer_report.md      # Tokenizer insights
├── model_card.md            # Model documentation
└── onepager.md              # Executive summary
```

### Logs
```
logs/
└── evaluation_TIMESTAMP.log  # Detailed execution log
```

## Understanding the Metrics

### F1 Score
- **Micro**: Aggregate across all labels (treats all predictions equally)
- **Macro**: Average of per-label F1 scores (treats all labels equally)
- **Range**: 0.0 to 1.0 (higher is better)

### ROC-AUC (Receiver Operating Characteristic - Area Under Curve)
- Measures discrimination ability across all thresholds
- **Range**: 0.0 to 1.0 (higher is better)
- Insensitive to class imbalance

### PR-AUC (Precision-Recall - Area Under Curve)
- Better for imbalanced datasets than ROC-AUC
- **Range**: 0.0 to 1.0 (higher is better)
- More informative for rare positive classes

### ECE (Expected Calibration Error)
- Measures calibration quality (confidence vs accuracy alignment)
- **Range**: 0.0 to 1.0 (lower is better)
- **Good calibration**: ECE < 0.05

### Brier Score
- Measures mean squared error between predictions and outcomes
- **Range**: 0.0 to 1.0 (lower is better)
- Combines calibration and discrimination

## Production Deployment

### Recommended Configuration

After evaluation, deploy with:

1. **Calibration Method**: Use the method recommended in `docs/onepager.md` (typically Temperature or Platt)

2. **Decision Thresholds**: Use per-label F1-optimal thresholds from `evaluation/thresholds_table.csv`

3. **Calibration Parameters**: Load from `evaluation/calibration_params.json`

### Example Deployment Code

```python
import json
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from scipy.special import expit as sigmoid

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained("BERT_Model/")
tokenizer = BertTokenizer.from_pretrained("BERT_Model/")
model.eval()

# Load calibration parameters
with open("evaluation/calibration_params.json") as f:
    cal_params = json.load(f)

# Get temperature and thresholds (example for temperature scaling)
temperature = cal_params['temperature']['T']
thresholds = cal_params['temperature']['thresholds']

def predict(text):
    """Predict toxicity with calibrated probabilities"""
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", 
                       max_length=256, truncation=True, padding=True)
    
    # Get logits
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.numpy()[0]
    
    # Apply temperature scaling
    calibrated_probs = sigmoid(logits / temperature)
    
    # Apply thresholds
    predictions = (calibrated_probs >= np.array(thresholds)).astype(int)
    
    return {
        'probabilities': calibrated_probs.tolist(),
        'predictions': predictions.tolist(),
        'labels': ['toxic', 'severe_toxic', 'obscene', 
                   'threat', 'insult', 'identity_hate']
    }

# Example usage
result = predict("This is a test comment")
print(result)
```

## Monitoring and Maintenance

### Regular Calibration Checks

Monitor these metrics on production data:

1. **ECE**: Should remain < 0.05
2. **Brier Score**: Should remain close to validation values
3. **F1 Score**: Monitor for performance degradation

### When to Recalibrate

Recalibrate if:
- ECE increases by > 0.02
- F1 score drops by > 5%
- Significant distribution shift detected
- New data becomes available

### Performance Monitoring

Track:
- Inference latency (target: < 100ms per batch of 32)
- Memory usage (target: < 1GB)
- Throughput (target: > 100 texts/sec on GPU)

## Troubleshooting

### Issue: Dependencies Not Found

```bash
# Solution: Set PYTHONPATH
export PYTHONPATH=/Users/seb/Desktop/EquityLens/equity-detector/venv/lib/python3.13/site-packages

# Or install in system Python
pip install -r requirements.txt
```

### Issue: CUDA Out of Memory

```bash
# Solution: Reduce batch size in scripts/run_evaluation_pipeline.py
# Change: BATCH_SIZE = 32 to BATCH_SIZE = 16 or BATCH_SIZE = 8
```

### Issue: Evaluation Takes Too Long

```bash
# Solution: Use GPU if available or reduce validation set size
# Edit run_tokenizer_and_summary.py:
# Change: max_samples=10000 to max_samples=5000
```

## FAQ

**Q: How long does the evaluation take?**
A: 30-60 minutes on CPU, 10-20 minutes on GPU

**Q: Can I use a different model?**
A: Yes, update MODEL_DIR in the scripts

**Q: How do I interpret the reliability diagrams?**
A: Perfect calibration = diagonal line. Deviations show over/under-confidence.

**Q: Which calibration method should I use?**
A: Check `docs/onepager.md` for the recommendation based on your data

**Q: Can I skip phases?**
A: Yes, but you'll miss corresponding visualizations and documentation

## References

- **BERT Paper**: [Devlin et al. (2018)](https://arxiv.org/abs/1810.04805)
- **Jigsaw Dataset**: [Kaggle Competition](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
- **Calibration Methods**: [Guo et al. (2017)](https://arxiv.org/abs/1706.04599)
- **Temperature Scaling**: [Original Paper](https://arxiv.org/abs/1706.04599)
- **Platt Scaling**: [Platt (1999)](https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf)

## Support

For issues or questions:
1. Check this guide
2. Review logs in `logs/`
3. Examine `docs/README.md` for detailed documentation
4. Check the main project repository

## License

This evaluation pipeline is part of the EquityLens project.

