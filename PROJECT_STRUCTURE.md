# EquityAlly Project Structure

**Last Updated:** October 24, 2025  
**Organization:** Cleaned and reorganized for clarity and maintainability

> **Note:** Major reorganization completed on October 24, 2025. See `equity-detector/REORGANIZATION_NOTES.md` for details on what changed and why.

---

## Overview

This document describes the complete directory structure of the EquityAlly project. The project has been organized into logical sections for better navigation and maintainability.

---

## Root Directory Structure

```
equityally/
├── equity-detector/           # Main application directory
├── video/                     # Diagrams used in README visuals
├── .gitignore                 # Git ignore patterns
├── LICENSE                    # Project license
├── PROJECT_STRUCTURE.md       # This file
└── README.md                  # Project overview
```

---

## Main Application: `equity-detector/`

The core Streamlit web application for toxic content detection.

### Application Files
```
equity-detector/
├── Home.py                   # Main entry point (Streamlit homepage)
├── requirements.txt          # Python dependencies
├── packages.txt              # System packages for deployment
└── .streamlit/config.toml    # Streamlit theme/config
```

### Pages (`pages/`)
Streamlit multi-page app structure:
```
pages/
├── 1_🔍_Detector.py         # Main detection interface (single + bulk analysis)
├── 2_📊_Performance.py       # Model performance metrics and visualizations
├── 3_📚_Learn_More.py        # Educational resources
└── 4_ℹ️_About.py            # About page and project information
```

### Utilities (`utils/`)
Core application logic and helper functions:
```
utils/
├── __init__.py              # Package initialization
├── bert_model.py            # BERT model loading and inference
├── openai_helper.py         # GPT-4o chat interface
└── shared.py                # Shared utilities and helpers
```

### Model (`BERT_Model/`)
Fine-tuned BERT model files:
```
BERT_Model/
├── config.json              # Model configuration
├── model.safetensors        # Model weights (418MB)
├── special_tokens_map.json  # Special tokens mapping
├── tokenizer_config.json    # Tokenizer configuration
└── vocab.txt                # BERT vocabulary
```

### Assets (`assets/`)
Visual assets, logos, diagrams, and sample CSVs:
```
assets/
├── README.md                # Assets documentation
├── equitylogo.svg           # Square logo
├── equitylogolong.svg       # Horizontal logo
├── diagrams/                # PNG diagram assets
└── sample_*.csv             # Sample CSVs used by the app
```

### Jigsaw Dataset (`data/training/jigsaw-toxic-comments/`)
Training dataset from Kaggle:
```
data/training/jigsaw-toxic-comments/
├── train.csv                # Training data
├── train.csv.zip            # Compressed training data
├── test.csv                 # Test data
├── test.csv.zip             # Compressed test data
├── test_labels.csv          # Test labels
└── test_labels.csv.zip      # Compressed test labels
```

### Data (`data/`)
Sample data and training datasets:
```
data/
├── README.md                # Data directory documentation
├── samples/                 # Sample CSV files for testing
│   ├── README.md
│   ├── sample_classroom.csv
│   ├── sample_data.csv
│   └── sample_social_media.csv
└── training/                # Training datasets
    ├── README.md
    └── jigsaw-toxic-comments/
        ├── train.csv        # Training data
        ├── train.csv.zip
        ├── test.csv         # Test data
        ├── test.csv.zip
        └── test_labels.csv  # Test labels
```

### Evaluation (`evaluation/`)
Model evaluation results and calibration artifacts:
```
evaluation/
├── calibration_params.json          # Calibration parameters
├── isotonic_calibration.json        # Isotonic regression model
├── metrics_summary.csv             # All metrics summary
├── thresholds_table.csv            # Decision threshold analysis
│
├── val_*.npy / test_*.npy          # Validation and test data
│   ├── val_labels.npy              # True labels
│   ├── val_logits.npy              # Model logits
│   ├── val_probs.npy               # Predicted probabilities
│   └── test_*.npy                   # Same for test set
│
└── metrics_*.json                   # Detailed metrics by method
    ├── metrics_val_uncal.json      # Uncalibrated validation
    ├── metrics_val_temperature.json # Temperature-scaled validation
    ├── metrics_val_isotonic.json   # Isotonic calibration validation
    └── metrics_test_*.json         # Same for test set
```

### Visualizations
Generated charts are produced during evaluation but are not tracked in git. Regenerate by running scripts in `scripts/`. For static visuals used in documentation, see `video/diagrams/` and `assets/diagrams/`.

### Scripts (`scripts/`)
Utility and automation scripts:
```
scripts/
├── calibration/                     # Calibration scripts
│   ├── README.md                    # Calibration scripts documentation
│   ├── fit_isotonic_calibration.py  # Calibration fitting script
│   ├── fit_isotonic_simple.py       # Simplified calibration script
│   └── test_calibration.py          # Calibration testing script
├── check_and_install_deps.py        # Dependency checker
├── comprehensive_test.py            # Full system test
├── diagnostic_check.py              # Diagnostics
├── generate_model_card.py           # Model card generator
├── run_all.sh                       # Run all evaluation scripts
├── run_complete_evaluation.py       # Complete evaluation pipeline
├── run_evaluation_pipeline.py       # Evaluation pipeline
├── run_tokenizer_and_summary.py     # Tokenizer analysis
├── run_with_logging.sh              # Run with logging
├── run_with_recording.sh            # Run with screen recording
├── monitor_progress.sh              # Progress monitor
└── show_implementation_summary.sh   # Show implementation details
```

### Documentation (`docs/`)
Comprehensive project documentation:
```
docs/
├── README.md                        # Documentation index
├── onepager.md                      # One-page project summary
├── model_card.md                    # ML model documentation
├── dataset_card.md                  # Dataset documentation
├── tokenizer_report.md              # Tokenizer analysis
│
├── calibration.md                   # Calibration methodology
├── OCR_GUIDE.md                    # OCR setup and usage
│
├── BULK_ANALYSIS_QUICKSTART.md     # Quick start guide
├── BULK_ANALYSIS_GUIDE.md          # Detailed bulk analysis guide
├── BULK_ANALYSIS_CHEATSHEET.md     # Quick reference
├── SAMPLE_FILES_README.md          # Sample data documentation
│
├── AI_CHAT_FEATURE.md              # AI chat feature docs
├── CHAT_INTERFACE_SUMMARY.md       # Chat interface summary
├── GPT_MODEL_UPDATE.md             # GPT integration update
├── WEBAPP_UPDATE_SUMMARY.md        # Web app updates
├── FILE_ORGANIZATION.md            # File organization notes
├── ORGANIZED_STRUCTURE.md          # Structure documentation
├── ISOTONIC_CALIBRATION_IMPLEMENTATION.md  # Calibration implementation
│
├── VIDEO_SCRIPT_COMPREHENSIVE.md   # Video script draft
│
├── evaluation_guides/              # Evaluation documentation
│   ├── EVALUATION_GUIDE.md
│   ├── EVALUATION_STATUS.md
│   ├── FAST_EVALUATION_SUMMARY.md
│   ├── FINAL_EVALUATION_STATUS.md
│   ├── PRE_RUN_CHECKLIST.md
│   └── ASCIINEMA_GUIDE.md
│
└── scripts_documentation/          # Scripts documentation
    └── README.md
```

### Recordings
Demo recordings are not tracked in the public repository.

### Logs
Log files are not tracked in the public repository.

---

## Video Assets: `video/`

Static diagram assets used in README visuals:

```
video/
└── diagrams/                        # PNG/Mermaid diagrams for documentation
```

---

## Project Documentation: `project-docs/`

High-level project summaries and update reports:

```
project-docs/
├── COMPLETE_PROJECT_SUMMARY.md              # Complete project overview
├── FINAL_DELIVERABLES_SUMMARY.md           # Deliverables checklist
├── IMPLEMENTATION_SUMMARY.md               # Implementation overview
├── IMPLEMENTATION_SUMMARY_DETAILED.md      # Detailed implementation
├── CALIBRATION_UPDATE_COMPLETE.md          # Calibration update report
├── CALIBRATION_UPDATE_SUMMARY.md           # Calibration summary
└── UPDATE_VERIFICATION_REPORT.md           # Verification report
```

---

## Key Files Quick Reference

### Essential for Running the App
- `equity-detector/Home.py` - Application entry point
- `equity-detector/requirements.txt` - Python dependencies
- `equity-detector/BERT_Model/` - Trained model files
- `equity-detector/utils/` - Core application logic

### Model Performance
- `equity-detector/evaluation/metrics_summary.csv` - All metrics
- `equity-detector/evaluation/isotonic_calibration.json` - Calibration model

### Documentation
- `equity-detector/docs/README.md` - Documentation index
- `equity-detector/docs/README.md` - Documentation index
- `equity-detector/docs/onepager.md` - Quick overview
- `PROJECT_STRUCTURE.md` - This file

### Video Production
- `video/narration.txt` - Final script
- `video/editors_track.txt` - Visual directions

---

## File Naming Conventions

### Documentation Files
- **UPPERCASE.md** - Project-level documentation and summaries
- **lowercase.md** - User-facing documentation and guides
- **CamelCase.md** - Feature-specific documentation

### Code Files
- **snake_case.py** - Python modules
- **PascalCase.py** - Streamlit pages (following Streamlit convention)

### Data Files
- **lowercase_description.csv** - Sample data files
- **snake_case.npy** - NumPy array files
- **snake_case.json** - Configuration and results

### Visualization Files
- **category_description_type.png** - Following pattern: `{split}_{method}_{chart_type}.png`
  - Example: `val_isotonic_reliability.png`

---

## Dependencies

### Python Packages
See `equity-detector/requirements.txt`:
- streamlit
- transformers
- torch
- pandas
- numpy
- matplotlib
- plotly
- scikit-learn
- openai (optional, for AI chat)

### System Packages
See `equity-detector/packages.txt` (for Streamlit Cloud deployment)

---

## Development Workflow

### Running the App Locally
```bash
cd equity-detector
streamlit run Home.py
```

### Running Evaluation
```bash
cd equity-detector/scripts
./run_all.sh
```

### Accessing Documentation
```bash
cd equity-detector/docs
open README.md
```

---

## Notes

- All paths are relative to the repository root (`equityally/`)
- The `equity-detector/` directory is the deployable Streamlit app
- Configuration files (`.streamlit/`, `.devcontainer/`) are in their respective directories
- Cache files (`__pycache__/`) are ignored by git
- The BERT model is 418MB and tracked with Git LFS (if enabled)

---

## Deployment

### Streamlit Cloud
- Root: `equity-detector/`
- Main file: `Home.py`
- Python version: 3.9+
- Dependencies: `requirements.txt`, `packages.txt`

### Local Development
```bash
git clone <repository>
cd equityally/equity-detector
pip install -r requirements.txt
streamlit run Home.py
```

---

## License & Attribution

See individual files for license information. Model trained on public datasets including:
- Jigsaw Toxic Comment Classification Challenge
- Wikipedia Talk Pages
- Twitter datasets
- Civil Comments

---

**Maintained by:** EquityAlly Development Team  
**Last Reorganization:** October 23, 2025

