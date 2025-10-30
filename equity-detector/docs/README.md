# EquityAlly Documentation Hub

**Last Updated:** October 24, 2025  
**Status:** Production Ready

---

## 🚀 Quick Start

### New to EquityAlly?
1. **[README.md](../../README.md)** - Start here for project overview
2. **[onepager.md](onepager.md)** - Executive summary and key results
3. **[model_card.md](model_card.md)** - Complete model documentation

### Ready to Deploy?
1. **[calibration.md](calibration.md)** - Calibration methodology
2. **[../evaluation/calibration_params.json](../evaluation/calibration_params.json)** - Production config
3. **[../evaluation/metrics_summary.csv](../evaluation/metrics_summary.csv)** - Performance metrics

---

## 📚 Documentation by Category

### Core Documentation

| Document | Description |
|----------|-------------|
| **[model_card.md](model_card.md)** | Complete model documentation, architecture, training |
| **[onepager.md](onepager.md)** | Executive summary with production recommendations |
| **[calibration.md](calibration.md)** | Calibration analysis and methodology |
| **[dataset_card.md](dataset_card.md)** | Dataset statistics, sources, and schema |
| **[tokenizer_report.md](tokenizer_report.md)** | Tokenizer analysis and insights |
| **[STRUCTURE.md](STRUCTURE.md)** | Project file organization guide |

### Feature Guides

| Document | Description |
|----------|-------------|
| **[BULK_ANALYSIS_GUIDE.md](BULK_ANALYSIS_GUIDE.md)** | Complete CSV/Excel bulk analysis tutorial |
| **[BULK_ANALYSIS_QUICKSTART.md](BULK_ANALYSIS_QUICKSTART.md)** | Quick start for bulk analysis |
| **[BULK_ANALYSIS_CHEATSHEET.md](BULK_ANALYSIS_CHEATSHEET.md)** | Quick reference for bulk analysis |
| **[SAMPLE_FILES_README.md](SAMPLE_FILES_README.md)** | Guide to sample data files |
| **[OCR_GUIDE.md](OCR_GUIDE.md)** | OCR setup and image text extraction |

### AI Features

| Document | Description |
|----------|-------------|
| **[AI_CHAT_FEATURE.md](AI_CHAT_FEATURE.md)** | AI chat assistant documentation |
| **[CHAT_INTERFACE_SUMMARY.md](CHAT_INTERFACE_SUMMARY.md)** | Chat interface overview |
| **[GPT_MODEL_UPDATE.md](GPT_MODEL_UPDATE.md)** | GPT integration updates |

### Technical Documentation

| Document | Description |
|----------|-------------|
| **[ISOTONIC_CALIBRATION_IMPLEMENTATION.md](ISOTONIC_CALIBRATION_IMPLEMENTATION.md)** | Detailed calibration implementation |
| **[WEBAPP_UPDATE_SUMMARY.md](WEBAPP_UPDATE_SUMMARY.md)** | Web application updates |
| **[VIDEO_SCRIPT_COMPREHENSIVE.md](VIDEO_SCRIPT_COMPREHENSIVE.md)** | Demo video script |

### Evaluation Guides

| Document | Description |
|----------|-------------|
| **[evaluation_guides/FINAL_EVALUATION_STATUS.md](evaluation_guides/FINAL_EVALUATION_STATUS.md)** | Latest evaluation status |
| **[evaluation_guides/EVALUATION_GUIDE.md](evaluation_guides/EVALUATION_GUIDE.md)** | Complete evaluation user manual |
| **[evaluation_guides/PRE_RUN_CHECKLIST.md](evaluation_guides/PRE_RUN_CHECKLIST.md)** | Pre-flight checklist for evaluation |
| **[evaluation_guides/ASCIINEMA_GUIDE.md](evaluation_guides/ASCIINEMA_GUIDE.md)** | Recording guide for demos |

### Scripts Documentation

| Document | Description |
|----------|-------------|
| **[scripts_documentation/README.md](scripts_documentation/README.md)** | Complete scripts reference |
| **[../scripts/calibration/README.md](../scripts/calibration/README.md)** | Calibration scripts guide |

---

## 🎯 Common Tasks

### I want to...

| Task | Go To |
|------|-------|
| **Understand the model** | [model_card.md](model_card.md) |
| **See evaluation results** | [onepager.md](onepager.md) |
| **Deploy to production** | [calibration.md](calibration.md) + [../evaluation/calibration_params.json](../evaluation/calibration_params.json) |
| **Analyze CSV files in bulk** | [BULK_ANALYSIS_GUIDE.md](BULK_ANALYSIS_GUIDE.md) |
| **Set up OCR for images** | [OCR_GUIDE.md](OCR_GUIDE.md) |
| **Run evaluation pipeline** | [evaluation_guides/EVALUATION_GUIDE.md](evaluation_guides/EVALUATION_GUIDE.md) |
| **Understand file structure** | [STRUCTURE.md](STRUCTURE.md) |
| **Learn about calibration** | [calibration.md](calibration.md) |
| **Check performance metrics** | [../evaluation/metrics_summary.csv](../evaluation/metrics_summary.csv) |

---

## 📊 Evaluation Results

All evaluation artifacts are in `../evaluation/`:

- **[metrics_summary.csv](../evaluation/metrics_summary.csv)** - All metrics across methods
- **[calibration_params.json](../evaluation/calibration_params.json)** - Production calibration config
- **[thresholds_table.csv](../evaluation/thresholds_table.csv)** - Optimal decision thresholds
- **[isotonic_calibration.json](../evaluation/isotonic_calibration.json)** - Isotonic calibration model

---

## 🖼️ Visuals

Static diagrams used in README and docs are available in:

- `../video/diagrams/` (PNG/Mermaid diagrams)
- `../assets/diagrams/` (PNG diagrams inside the app)

Evaluation visualizations are generated locally when you run the evaluation scripts and are not tracked in this repository.

---

## 🔧 Reproducibility

### Run Complete Evaluation
```bash
cd ../scripts
python run_complete_evaluation.py
```

### Run Individual Components
```bash
# Evaluation pipeline
python run_evaluation_pipeline.py

# Tokenizer analysis
python run_tokenizer_and_summary.py

# Calibration fitting
cd calibration
python fit_isotonic_calibration.py
```

---

## 🎯 Performance Highlights

- **Validation Set**: 96.84% accuracy, 84.87% F1, 96.80% ROC-AUC
- **Test Set**: 91.52% accuracy, 67.19% F1, 96.80% ROC-AUC
- **Calibration Method**: Isotonic regression (best performance)
- **Decision Threshold**: 0.40 (F1-optimal for the calibrated model)
- **Inference Time**: <100ms on CPU

---

## ⚠️ Deprecated Files

Legacy and duplicate documentation has been moved to `../_deprecated/`:

- Old organization documents
- Superseded configuration files
- Outdated documentation versions

**Do not use files from `_deprecated/`** - they are kept for historical reference only.

For current documentation, use the files listed in this index.

---

## 📖 Additional Resources

### External Links
- **Main README**: [../../README.md](../../README.md)
- **Project Structure**: [../../PROJECT_STRUCTURE.md](../../PROJECT_STRUCTURE.md)
- **Reorganization Notes**: [../REORGANIZATION_NOTES.md](../REORGANIZATION_NOTES.md)

### Data Files
- **Sample Data**: [../data/samples/](../data/samples/)
- **Training Data**: [../data/training/jigsaw-toxic-comments/](../data/training/jigsaw-toxic-comments/)

### Application Files
- **Streamlit Pages**: [../pages/](../pages/)
- **Utilities**: [../utils/](../utils/)
- **Scripts**: [../scripts/](../scripts/)

---

## 🤝 Contributing

When adding new documentation:
1. Update this index with the new file
2. Follow the existing naming conventions
3. Add a one-line description
4. Place in the appropriate category
5. Update cross-references in related docs

---

## 📞 Support

For questions or issues:
1. Check the relevant documentation above
2. Review [model_card.md](model_card.md) for technical details
3. See [evaluation_guides/EVALUATION_GUIDE.md](evaluation_guides/EVALUATION_GUIDE.md) for troubleshooting
4. Refer to the main project repository

---

**Maintained by:** EquityAlly Development Team  
**Documentation Version:** 2.0  
**Last Major Update:** October 24, 2025
