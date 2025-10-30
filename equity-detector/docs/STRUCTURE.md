# EquityLens - Organized File Structure

**Organization Date**: October 19, 2025  
**Status**: ✅ Complete and Professional

---

## 📂 Clean Directory Structure

```
equity-detector/
│
├── 📄 README.md                        ⭐ Start here!
├── 📄 FILE_ORGANIZATION.md             Organization guide
├── 📄 requirements.txt                  Python dependencies
│
├── 🤖 BERT_Model/                      Pre-trained model (450 MB)
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer_config.json
│   └── vocab.txt
│
├── 📚 docs/                            All Documentation (20+ files)
│   ├── 📄 README.md                    Documentation hub
│   ├── ⭐ model_card.md                Complete model documentation
│   ├── ⭐ onepager.md                  Executive summary
│   ├── ⭐ calibration.md               Calibration analysis
│   ├── dataset_card.md                 Dataset statistics
│   ├── tokenizer_report.md             Tokenizer insights
│   │
│   ├── 📁 evaluation_guides/           Evaluation documentation (6 files)
│   │   ├── EVALUATION_GUIDE.md         Complete user manual
│   │   ├── ASCIINEMA_GUIDE.md          Recording guide
│   │   ├── PRE_RUN_CHECKLIST.md        Pre-flight checklist
│   │   └── ...
│   │
│   ├── 📁 scripts_documentation/       Scripts reference
│   │   └── README.md                   Complete scripts guide
│   │
│   └── BULK_ANALYSIS_*.md              Bulk analysis guides (3 files)
│       VIDEO_SCRIPT_COMPREHENSIVE.md
│       OCR_GUIDE.md
│
├── 📊 evaluation/                      Evaluation Artifacts (11 files)
│   ├── ⭐ calibration_params.json      Production config
│   ├── ⭐ thresholds_table.csv         Optimal thresholds
│   ├── metrics_summary.csv             All metrics
│   ├── metrics_*.json                  Detailed metrics
│   ├── val_logits.npy                  Raw predictions
│   ├── val_probs.npy
│   ├── test_logits.npy
│   ├── test_probs.npy
│   └── *_labels.npy
│
├── 🖼️ assets/                          Logos, diagrams, sample CSVs
│   ├── equitylogo.svg
│   ├── equitylogolong.svg
│   ├── diagrams/                       PNG diagram assets
│   └── sample_*.csv
│
├── 🔧 scripts/                         All Scripts (13 files)
│   ├── ⭐ run_complete_evaluation.py   Main pipeline
│   ├── run_evaluation_pipeline.py      Core evaluation
│   ├── run_tokenizer_and_summary.py    Tokenizer analysis
│   ├── generate_model_card.py
│   ├── diagnostic_check.py             Pre-flight tests
│   ├── comprehensive_test.py           Full test suite
│   ├── check_and_install_deps.py
│   │
│   ├── run_all.sh                      Shell automation
│   ├── run_with_logging.sh
│   ├── run_with_recording.sh
│   ├── monitor_progress.sh
│   └── show_implementation_summary.sh
│
├── 📋 logs/                            (not tracked in repo)
│
├── 🎬 recordings/                      (not tracked in repo)
│
├── 📊 pages/                           Streamlit App Pages
│   ├── 1_🔍_Detector.py
│   ├── 2_📊_Performance.py
│   ├── 3_📚_Learn_More.py
│   └── 4_ℹ️_About.py
│
├── 🛠️ utils/                           Utility Modules
│   ├── __init__.py
│   ├── bert_model.py
│   ├── openai_helper.py
│   └── shared.py
│
├── Home.py                             Streamlit main app
├── sample_*.csv                        Sample data (3 files)
├── equitylogo.svg
└── equitylogolong.svg
```

---

## 🎯 Quick Navigation

### I want to...

| Task | File/Folder |
|------|-------------|
| **Understand the model** | `docs/model_card.md` |
| **See evaluation results** | `docs/onepager.md` |
| **View static diagrams** | `../video/diagrams/` |
| **Deploy to production** | `evaluation/calibration_params.json` + `docs/model_card.md` |
| **Run evaluation again** | `scripts/run_complete_evaluation.py` |
| **Check metrics** | `evaluation/metrics_summary.csv` |
| **Replay evaluation** | `asciinema play recordings/bert_final.cast` |
| **Learn about calibration** | `docs/calibration.md` |
| **Understand organization** | `STRUCTURE.md` |
| **Use Streamlit app** | `streamlit run Home.py` |

---

## 📊 File Counts by Category

| Category | Location | Count |
|----------|----------|-------|
| **Documentation** | `docs/` | 20+ files |
| **Assets (diagrams, samples)** | `assets/` | multiple |
| **- Baseline** | `visualizations/baseline/` | 8 files |
| **- Calibrated** | `visualizations/calibrated/` | 24 files |
| **- Analysis** | `visualizations/analysis/` | 9 files |
| **- Dataset** | `visualizations/` (root) | 3 files |
| **Evaluation Data** | `evaluation/` | 11 files |
| **Scripts** | `scripts/` | 13 files |
| **Logs** | `logs/` | 1+ files |
| **Recordings** | `recordings/` | 1 file |

---

## 🌟 Key Files (Must Review)

### Top Priority ⭐
1. **docs/onepager.md** - Executive summary with recommendations
2. **docs/model_card.md** - Complete model documentation
3. **evaluation/calibration_params.json** - Production configuration
4. **docs/onepager.md** - Calibration proof visuals

### High Priority
5. **docs/calibration.md** - Calibration analysis
6. **evaluation/metrics_summary.csv** - All metrics
7. (reserved)
8. (reserved)

### Reference
9. **FILE_ORGANIZATION.md** - Organization guide
10. **docs/README.md** - Documentation hub

---

## 🧹 Cleanup Performed

### Removed
- ❌ Temporary log files (`evaluation_log.txt`)
- ❌ Old evaluation artifacts
- ❌ Duplicate files

### Organized
- ✅ Moved evaluation guides to `docs/evaluation_guides/`
- ✅ Moved scripts docs to `docs/scripts_documentation/`
- ✅ Categorized visualizations into subdirectories
- ✅ Moved bulk analysis guides to `docs/`
- ✅ Consolidated all documentation

### Structure
- ✅ Created logical folder hierarchy
- ✅ Separated baseline vs calibrated results
- ✅ Grouped analysis plots together
- ✅ Reserved summary/ for future dashboards

---

## 💾 Storage Requirements

| Component | Size |
|-----------|------|
| Model | ~450 MB |
| Dataset | ~130 MB |
| Visualizations | ~10 MB |
| Evaluation Data | ~15 MB |
| Documentation | ~500 KB |
| Scripts | ~200 KB |
| Recording | ~20 KB |
| **Total** | **~600 MB** |

---

## ✅ Organization Benefits

1. **Easy Navigation**: Clear folder structure
2. **Quick Access**: Key files marked with ⭐
3. **Professional**: Ready for sharing/publication
4. **Scalable**: Easy to add new evaluations
5. **Documented**: Every section explained
6. **Reproducible**: All artifacts preserved

---

**All files organized and ready for use!** 🎉

