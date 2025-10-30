# Pre-Run Checklist - BERT Evaluation Pipeline

**Date**: 2025-10-19  
**Status**: ✅ READY TO RUN

---

## ✅ Diagnostic Results

### Model Configuration
- ✅ Model: BertForSequenceClassification
- ✅ Architecture: BERT-base (110M params)
- ✅ Output Classes: 2 (binary classification)
- ✅ Task: Toxic vs Non-Toxic detection
- ✅ Vocab Size: 30,522 tokens

### Data Configuration
- ✅ Validation Set: 159,571 available → 10,000 sampled
- ✅ Test Set: 63,978 available → 5,000 sampled
- ✅ Text Column: `comment_text`
- ✅ Label: `toxic` (binary 0/1)
- ✅ Label Prevalence: ~9.4% toxic (balanced enough)

### Script Configuration
- ✅ All 3 scripts updated for binary mode
- ✅ LABEL_COLS = ['toxic'] (matches model)
- ✅ Visualizations updated for single label
- ✅ Batch size: 64 (optimized)
- ✅ Workers: 8 (parallel loading)
- ✅ Max length: 256 tokens

### File Tree Status
- ✅ visualizations/ - CLEAN (empty)
- ✅ evaluation/ - CLEAN (empty)
- ✅ logs/ - CLEAN (empty)
- ✅ recordings/ - CLEAN (empty)
- ✅ No cached files (.pyc, __pycache__)

---

## 📊 Expected Outputs

### Visualizations (21 total)

**Dataset (3 plots)**:
- label_prevalence_val.png
- label_prevalence_test.png
- class_imbalance_heatmap.png

**Baseline - Uncalibrated (4 plots)**:
- val_uncal_roc_perlabel.png
- test_uncal_roc_perlabel.png
- val_uncal_pr_perlabel.png
- test_uncal_pr_perlabel.png
- val_uncal_reliability.png
- test_uncal_reliability.png
- val_uncal_cm_aggregate.png
- test_uncal_cm_aggregate.png

**Temperature Scaling (4 plots)**:
- val_temp_roc_perlabel.png
- test_temp_roc_perlabel.png
- val_temp_pr_perlabel.png
- test_temp_pr_perlabel.png
- val_temp_reliability.png (🌟 KEY: shows calibration improvement)
- test_temp_reliability.png (🌟 KEY: shows calibration improvement)
- val_temp_cm_aggregate.png
- test_temp_cm_aggregate.png

**Platt Scaling (4 plots)**:
- val_platt_roc_perlabel.png
- test_platt_roc_perlabel.png
- val_platt_pr_perlabel.png
- test_platt_pr_perlabel.png
- val_platt_reliability.png
- test_platt_reliability.png
- val_platt_cm_aggregate.png
- test_platt_cm_aggregate.png

**Isotonic Regression (4 plots)**:
- val_isotonic_roc_perlabel.png
- test_isotonic_roc_perlabel.png
- val_isotonic_pr_perlabel.png
- test_isotonic_pr_perlabel.png
- val_isotonic_reliability.png
- test_isotonic_reliability.png
- val_isotonic_cm_aggregate.png
- test_isotonic_cm_aggregate.png

**Threshold Analysis (2 plots)**:
- threshold_sweep_toxic.png
- threshold_heatmap_all_methods.png

**Tokenizer Analysis (5+ plots)**:
- token_length_hist_val.png
- token_length_hist_test.png
- token_fragmentation_box_val.png
- token_fragmentation_box_test.png
- token_zipf_val.png
- token_zipf_test.png
- token_rare_topk_val.png

**Summary Dashboards (4 plots)**:
- summary_roc_overlay.png
- summary_pr_overlay.png
- summary_reliability_overlay.png
- summary_metrics_bars.png

**Total**: ~35-40 visualizations

### Evaluation Artifacts

- val_logits.npy
- val_probs.npy
- val_labels.npy
- test_logits.npy
- test_probs.npy
- test_labels.npy
- metrics_val_uncal.json
- metrics_test_uncal.json
- calibration_params.json
- thresholds_table.csv
- metrics_summary.csv

### Documentation

- docs/dataset_card.md
- docs/calibration.md
- docs/tokenizer_report.md
- docs/model_card.md ✅ (already created)
- docs/onepager.md
- docs/README.md

---

## 🎯 Calibration Showcase

### What Will Be Demonstrated

#### Uncalibrated (Baseline):
```
Metrics:
  F1 Score:    ~0.75-0.85
  ROC-AUC:     ~0.93-0.97
  ECE:         ~0.08-0.15 ⚠️ (POOR CALIBRATION)
  Brier:       ~0.08-0.12

Interpretation: Good discrimination but poor probability estimates
```

#### Post-Calibration (Temperature):
```
Metrics:
  F1 Score:    ~0.75-0.85 (maintained)
  ROC-AUC:     ~0.93-0.97 (maintained)
  ECE:         ~0.02-0.05 ✅ (EXCELLENT CALIBRATION)
  Brier:       ~0.06-0.09 (improved)

Improvement: 50-70% ECE reduction!
```

#### Post-Calibration (Platt):
```
Similar improvements to Temperature
```

#### Post-Calibration (Isotonic):
```
Similar or better improvements
```

### Key Visualizations

1. **Reliability Diagrams** 🌟
   - Uncalibrated: Points far from diagonal (poor calibration)
   - Calibrated: Points close to diagonal (good calibration)
   - Visual proof of calibration improvement

2. **ROC Curves**
   - Should be nearly identical (discrimination preserved)

3. **PR Curves**
   - Should be nearly identical (precision/recall preserved)

4. **Metrics Bar Chart**
   - Shows ECE improvement across methods
   - F1/AUC maintained

---

## ⏱️ Expected Timeline

| Phase | Task | Time | Cumulative |
|-------|------|------|------------|
| 1 | Data loading | 15 sec | 0:15 |
| 2 | Model loading | 10 sec | 0:25 |
| 3 | Validation inference | 6 min | 6:25 |
| 3 | Test inference | 3 min | 9:25 |
| 4 | Baseline metrics & viz | 1 min | 10:25 |
| 5 | Temperature calibration | 30 sec | 10:55 |
| 5 | Platt calibration | 30 sec | 11:25 |
| 5 | Isotonic calibration | 30 sec | 11:55 |
| 5 | Calibration viz | 2 min | 13:55 |
| 6 | Threshold optimization | 1 min | 14:55 |
| 7 | Tokenizer analysis | 3 min | 17:55 |
| 8 | Summary dashboards | 1 min | 18:55 |
| 9-10 | Documentation | 1 min | 19:55 |

**Total**: ~20 minutes

---

## 🎬 Recording Configuration

- ✅ asciinema installed
- ✅ Idle time limit: 2 seconds
- ✅ Title: "BERT Toxicity Evaluation - Calibration Showcase"
- ✅ Dual output: console + log file
- ✅ Recording file: `recordings/evaluation_*.cast`

---

## 🚀 Performance Optimizations

| Optimization | Value | Impact |
|--------------|-------|--------|
| Batch Size | 64 | 2x throughput |
| Workers | 8 | Parallel loading |
| Val Samples | 10,000 | 16x faster |
| Test Samples | 5,000 | 13x faster |
| Tokenizer Samples | 2,000/1,000 | 5x faster |

**Overall Speedup**: ~15x (3-4 hours → 20 minutes)

---

## 🔍 Final Pre-Run Checks

### Environment
- [x] Python 3.13.5
- [x] PyTorch 2.9.0
- [x] Transformers 4.53.0
- [x] All dependencies installed
- [x] PYTHONPATH configured

### Model & Data
- [x] Model loads successfully
- [x] Test inference works (2-class output)
- [x] Data files accessible
- [x] Column names verified

### Scripts
- [x] Binary mode enabled (LABEL_COLS = ['toxic'])
- [x] Visualizations handle single label
- [x] Metrics handle single label
- [x] Calibration methods ready
- [x] No syntax errors

### File System
- [x] Directories exist (visualizations, evaluation, logs, recordings, docs)
- [x] All directories clean (no old artifacts)
- [x] Sufficient disk space (~1 GB free)

---

## ⚡ Ready to Execute

**Command**:
```bash
cd /Users/seb/Desktop/EquityLens/equity-detector

PYTHONPATH=/Users/seb/Desktop/EquityLens/equity-detector/venv/lib/python3.13/site-packages \
asciinema rec \
  --overwrite \
  --idle-time-limit 2 \
  --title "BERT Toxicity Evaluation - Calibration Showcase" \
  recordings/bert_calibration_evaluation.cast \
  --command "python3 scripts/run_complete_evaluation.py 2>&1 | tee logs/evaluation_$(date +%Y%m%d_%H%M%S).log" &
```

---

## 📋 Post-Run Actions

After completion (~20 min):

1. ✅ Review executive summary in console/log
2. ✅ Check `docs/onepager.md` for recommendations
3. ✅ View calibration improvements in `visualizations/*_reliability.png`
4. ✅ Replay recording: `asciinema play recordings/bert_calibration_evaluation.cast`
5. ✅ Load production config: `evaluation/calibration_params.json`

---

## 🎯 Success Criteria

- [ ] All 3 calibration methods evaluated
- [ ] ECE reduction of >40% demonstrated
- [ ] F1/AUC maintained (±0.02)
- [ ] ~35-40 visualizations generated
- [ ] 6 documentation files created
- [ ] Production config saved
- [ ] asciinema recording captured
- [ ] Executive summary displayed

---

**Status**: ✅ ALL CHECKS PASSED - READY TO RUN! 🚀

