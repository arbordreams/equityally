# Evaluation Scripts

This directory contains all scripts for running the BERT toxicity model evaluation pipeline.

## Scripts Overview

### Main Execution Scripts

#### `run_complete_evaluation.py`
Complete end-to-end evaluation pipeline (Phases 1-10).

**Usage**:
```bash
PYTHONPATH=../venv/lib/python3.13/site-packages python3 run_complete_evaluation.py
```

**What it does**:
- Runs all evaluation phases
- Generates all visualizations
- Creates all documentation
- Produces executive summary

**Runtime**: ~30-60 minutes

---

#### `run_evaluation_pipeline.py`
Core evaluation pipeline (Phases 1-6): data loading, inference, calibration, and baseline visualization.

**Usage**:
```bash
PYTHONPATH=../venv/lib/python3.13/site-packages python3 run_evaluation_pipeline.py
```

**What it does**:
- Loads data and model
- Runs uncalibrated inference
- Fits calibration methods (Temperature, Platt, Isotonic)
- Generates ROC, PR, reliability diagrams
- Optimizes decision thresholds

**Outputs**:
- `evaluation/*.npy` - Raw logits and probabilities
- `evaluation/metrics_*.json` - Metrics for each method
- `visualizations/*_roc_*.png` - ROC curves
- `visualizations/*_pr_*.png` - PR curves
- `visualizations/*_reliability.png` - Calibration plots

---

#### `run_tokenizer_and_summary.py`
Tokenizer analysis and summary visualizations (Phase 7).

**Usage**:
```bash
PYTHONPATH=../venv/lib/python3.13/site-packages python3 run_tokenizer_and_summary.py
```

**What it does**:
- Analyzes token distributions
- Computes fragmentation metrics
- Generates Zipf plots
- Identifies rare/toxic tokens
- Creates summary dashboards
- Generates documentation

**Outputs**:
- `visualizations/token_*.png` - Token analysis plots
- `visualizations/summary_*.png` - Comparative dashboards
- `docs/tokenizer_report.md` - Detailed findings
- `docs/dataset_card.md` - Dataset statistics

---

### Utility Scripts

#### `check_and_install_deps.py`
Dependency checker and installation helper.

**Usage**:
```bash
python3 check_and_install_deps.py
```

**What it does**:
- Checks for required packages
- Reports missing dependencies
- Provides installation commands

---

### Shell Wrappers

#### `run_all.sh`
Simple shell wrapper for the complete evaluation.

**Usage**:
```bash
bash run_all.sh
```

**Features**:
- Activates virtual environment (if needed)
- Checks dependencies
- Runs complete evaluation
- Reports results locations

---

#### `run_with_logging.sh`
Runs evaluation with comprehensive logging.

**Usage**:
```bash
bash run_with_logging.sh
```

**Features**:
- Creates timestamped log file in `logs/`
- Captures stdout and stderr
- Displays output to console and file simultaneously

**Log location**: `logs/evaluation_YYYYMMDD_HHMMSS.log`

---

#### `run_with_recording.sh`
Runs evaluation with asciinema terminal recording.

**Usage**:
```bash
bash run_with_recording.sh
```

**Features**:
- Records terminal session for documentation
- Creates `.cast` file for replay
- Can be uploaded to asciinema.org for sharing

**Requirements**: `brew install asciinema`

**Recording location**: `recordings/complete_evaluation.cast`

**Replay**:
```bash
asciinema play recordings/complete_evaluation.cast
```

---

## Configuration

All scripts use these constants (defined in each script):

```python
BASE_DIR = Path("/Users/seb/Desktop/EquityLens/equity-detector")
MODEL_DIR = BASE_DIR / "BERT_Model"
DATA_DIR = BASE_DIR / "data/training/jigsaw-toxic-comments"
VIZ_DIR = BASE_DIR / "visualizations"
DOCS_DIR = BASE_DIR / "docs"
EVAL_DIR = BASE_DIR / "evaluation"

LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
TEXT_COL = "comment_text"
MAX_LEN = 256
BATCH_SIZE = 32
RANDOM_SEED = 42
```

To modify:
- Edit the constants at the top of each script
- Or use environment variables (future enhancement)

---

## Script Dependencies

```
run_complete_evaluation.py
├── run_evaluation_pipeline.py (Phases 1-6)
└── run_tokenizer_and_summary.py (Phase 7)
```

You can run scripts individually if you only need specific phases.

---

## Output Structure

After running any main script:

```
equity-detector/
├── visualizations/          # All plots and charts
│   ├── label_prevalence_*.png
│   ├── *_roc_perlabel.png
│   ├── *_pr_perlabel.png
│   ├── *_reliability.png
│   ├── *_cm_aggregate.png
│   ├── threshold_sweep_*.png
│   ├── token_*.png
│   └── summary_*.png
│
├── evaluation/              # Metrics and artifacts
│   ├── *.npy               # Raw predictions
│   ├── metrics_*.json      # Metrics per method
│   ├── calibration_params.json
│   ├── thresholds_table.csv
│   └── metrics_summary.csv
│
├── docs/                    # Documentation
│   ├── README.md
│   ├── dataset_card.md
│   ├── calibration.md
│   ├── tokenizer_report.md
│   ├── model_card.md
│   └── onepager.md
│
└── logs/                    # Execution logs
    └── evaluation_*.log
```

---

## Customization

### Change Batch Size

Edit `BATCH_SIZE` in the scripts:

```python
# For GPU with limited memory
BATCH_SIZE = 16  # Default: 32

# For CPU or debugging
BATCH_SIZE = 8
```

### Change Max Sequence Length

Edit `MAX_LEN` in the scripts:

```python
# For longer texts
MAX_LEN = 512  # Default: 256

# Note: Increases memory and compute time
```

### Sample Fewer Texts for Tokenizer Analysis

Edit `run_tokenizer_and_summary.py`:

```python
# Line ~280
val_stats = analyze_tokenization(val_df, tokenizer, "Validation", max_samples=5000)
# Default: 10000
```

### Use GPU

The scripts automatically detect and use GPU if available:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

To force CPU:
```python
device = torch.device("cpu")
```

---

## Troubleshooting

### ImportError: No module named 'X'

```bash
# Check dependencies
python3 check_and_install_deps.py

# Install missing packages
pip install -r ../requirements.txt
```

### PYTHONPATH Issues

```bash
# Set explicitly before running
export PYTHONPATH=/Users/seb/Desktop/EquityLens/equity-detector/venv/lib/python3.13/site-packages
python3 run_complete_evaluation.py
```

### Out of Memory

```bash
# Reduce batch size in scripts
# Edit BATCH_SIZE = 16 or BATCH_SIZE = 8
```

### FileNotFoundError

```bash
# Ensure you're in the correct directory
cd /Users/seb/Desktop/EquityLens/equity-detector

# Ensure data files are unzipped
ls data/training/jigsaw-toxic-comments/*.csv
# Should see: train.csv, test.csv, test_labels.csv
```

---

## Performance Tips

### For Faster Execution

1. **Use GPU**: Significant speedup (5-10x faster)
2. **Increase Batch Size**: If you have GPU memory
3. **Reduce Validation Samples**: For tokenizer analysis
4. **Skip Optional Visualizations**: Comment out UMAP/t-SNE sections

### For Lower Memory Usage

1. **Decrease Batch Size**: `BATCH_SIZE = 8`
2. **Process in Chunks**: Modify inference loop
3. **Delete Intermediate Files**: After calibration completes

---

## Development

### Adding New Calibration Methods

Edit `run_evaluation_pipeline.py`:

1. Create calibration class (similar to `TemperatureScaling`)
2. Add to calibration phase (after line ~800)
3. Generate visualizations
4. Update documentation

### Adding New Metrics

Edit `compute_all_metrics()` function:

```python
def compute_all_metrics(y_true, y_pred_probs, thresholds=None):
    # Add your metric here
    metrics["custom_metric"] = your_metric_function(y_true, y_pred_probs)
    return metrics
```

### Adding New Visualizations

Add plotting functions following the pattern:

```python
def plot_your_visualization(data, save_path):
    fig, ax = plt.subplots(figsize=(12, 6))
    # Your plotting code
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved: {save_path}")
```

---

## Version History

- **v1.0** (2025-10-19): Initial complete evaluation pipeline
  - All calibration methods implemented
  - Comprehensive visualization suite
  - Full documentation generation
  - Executive summary output

---

## Contact

For questions or issues:
1. Review the main `EVALUATION_GUIDE.md`
2. Check execution logs in `logs/`
3. Examine script outputs for error messages

---

## Quick Reference

| Task | Command |
|------|---------|
| Check dependencies | `python3 check_and_install_deps.py` |
| Run complete pipeline | `bash run_with_logging.sh` |
| Run with recording | `bash run_with_recording.sh` |
| Run only inference | `python3 run_evaluation_pipeline.py` |
| Run only tokenizer | `python3 run_tokenizer_and_summary.py` |
| View results | `open ../docs/onepager.md` |
| Check logs | `tail -f ../logs/evaluation_*.log` |

