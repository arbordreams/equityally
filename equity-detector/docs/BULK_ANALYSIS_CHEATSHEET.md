# 📊 Bulk Analysis Feature - Quick Reference Card

## 🚀 3-Step Quick Start

```
1. SELECT MODE → "📊 Upload CSV/Bulk Analysis"
2. UPLOAD FILE → Choose CSV/Excel → Select text column
3. CLICK ANALYZE → Wait for results → Review & Download
```

---

## 📁 File Requirements

| Format | Extension | Library |
|--------|-----------|---------|
| CSV | `.csv` | pandas (built-in) |
| Excel | `.xlsx` | openpyxl (installed) |
| Excel Legacy | `.xls` | openpyxl (installed) |

**Minimum CSV:**
```csv
text
"Your content here"
"Another entry"
```

**Recommended CSV:**
```csv
id,text,author
1,"Content","User123"
2,"More content","User456"
```

---

## 📊 4 Visualizations You Get

| Chart | What It Shows | Use For |
|-------|---------------|---------|
| 🥧 **Pie Chart** | Safe vs. Concerning split | Overall distribution |
| 📊 **Histogram** | Risk score distribution | Identifying patterns |
| 📦 **Box Plot** | Statistical comparison | Understanding variance |
| 📈 **Bar Chart** | Low/Moderate/High severity | Risk level breakdown |

---

## 🎯 Key Metrics Displayed

```
┌─────────────────────────────────────────┐
│  Total Analyzed:    100                 │
│  Safe Content:      65 (65%)            │
│  Concerning:        35 (35%)            │
│  Avg Risk Score:    38.5%               │
└─────────────────────────────────────────┘
```

---

## 🔍 Filter & Sort Options

### Filters
- ☑️ **All** - Show everything
- ✅ **Safe Only** - Non-concerning content
- ⚠️ **Concerning Only** - Flagged content
- 🔴 **High Risk Only** - >70% risk score

### Sorting
- 📋 **Original Order** - As uploaded
- ⬆️ **High to Low** - Most risky first
- ⬇️ **Low to High** - Least risky first

---

## 💾 Download Options

### Option 1: CSV Results
```
Filename: equity_analysis_results_20241019_143022.csv
Contains: Full text, scores, classifications, IDs
```

### Option 2: Summary Report
```
Filename: equity_analysis_summary_20241019_143022.txt
Contains: Stats, percentages, timestamps, config
```

---

## ⏱️ Processing Speed Reference

| Entries | Standard Mode | MC Dropout (20x) |
|---------|---------------|------------------|
| 10      | 1-3 sec       | 20-60 sec        |
| 50      | 5-15 sec      | 2-5 min          |
| 100     | 10-30 sec     | 3-10 min         |
| 500     | 1-3 min       | 15-50 min        |
| 1000    | 2-5 min       | 30-100 min       |

**💡 Tip:** Start without MC Dropout, then re-analyze high-risk entries with it enabled.

---

## 🎲 Risk Score Interpretation

```
0-50%    │ ████████████░░░░░░░░ │ LOW      │ ✅ Generally safe
30-70%   │ ████████████████░░░░ │ MODERATE │ ⚡ Review recommended
80-90%   │ ██████████████████░░ │ HIGH     │ ⚠️ Immediate review
90-100%  │ ████████████████████ │ CRITICAL │ 🔴 Urgent action
         └──────────────────────┘
         Decision Threshold: 40% (Isotonic calibrated)
```

---

## 💡 AI-Generated Insights

You automatically get:
- ✅ **Overall Assessment** - High Alert / Moderate / Positive
- 🔴 **High-Risk Count** - Entries >70% requiring attention
- 🟡 **Moderate-Risk Count** - Borderline entries (47-70%)
- 🎲 **Uncertainty Warnings** - If MC Dropout shows high variance

---

## 🧪 Sample Files Provided

| File | Entries | Purpose |
|------|---------|---------|
| `sample_data.csv` | 20 | General testing |
| `sample_social_media.csv` | 15 | Forum/social context |
| `sample_classroom.csv` | 15 | Educational context |

**Try them!** Upload → Select "text"/"comment"/"message" column → Analyze

---

## 🔧 Common Issues & Fixes

| Problem | Solution |
|---------|----------|
| 🚫 Upload fails | Check file format (CSV/XLSX/XLS only) |
| ⚠️ No valid entries | Ensure text >5 characters, select correct column |
| 🐌 Too slow | Disable MC Dropout, split into smaller batches |
| 📊 Charts missing | Check browser compatibility, refresh page |
| 💾 Download fails | Check browser permissions, try different browser |

---

## ⚙️ Advanced: Monte Carlo Dropout

```
┌──────────────────────────────────────────────┐
│  Enable: Advanced Options → ☑ MC Dropout     │
│  Passes: 20-30 recommended                   │
│  Benefit: +3-5% accuracy, uncertainty metrics│
│  Cost: 10-30x slower processing              │
└──────────────────────────────────────────────┘
```

**When to use:**
- Critical applications
- High-stakes decisions
- Research requiring confidence intervals
- Borderline cases needing extra scrutiny

---

## 📚 Documentation Quick Links

| Resource | Purpose |
|----------|---------|
| [BULK_ANALYSIS_GUIDE.md](BULK_ANALYSIS_GUIDE.md) | Complete documentation |
| [BULK_ANALYSIS_QUICKSTART.md](BULK_ANALYSIS_QUICKSTART.md) | Fast learning guide |
| [SAMPLE_FILES_README.md](SAMPLE_FILES_README.md) | Sample file info |
| [README.md](README.md) | Main project docs |

---

## 🎯 Typical Workflows

### Workflow 1: Quick Review (5 min)
```
1. Upload CSV
2. Select columns
3. Analyze
4. Filter "High Risk Only"
5. Review flagged content
6. Done!
```

### Workflow 2: Comprehensive Analysis (15 min)
```
1. Upload CSV
2. Enable MC Dropout (20 passes)
3. Analyze with uncertainty
4. Review all visualizations
5. Download full results
6. Generate report for stakeholders
```

### Workflow 3: Ongoing Moderation (Weekly)
```
1. Export week's content
2. Analyze in bulk
3. Sort by risk score
4. Moderate top 10% manually
5. Archive results
6. Track trends over time
```

---

## 🔒 Privacy & Security

```
✅ All BERT analysis runs locally (on your device)
✅ No data sent to external servers
✅ Results only saved if you download
✅ No cloud storage required
✅ OpenAI features are optional (require API key)
```

**Your data stays YOUR data!**

---

## 📊 Expected Results (Sample Files)

```
sample_data.csv:
├─ 50% concerning (expected)
├─ ~45-55% avg risk score
└─ Good for testing all features

sample_social_media.csv:
├─ 40-50% concerning (typical)
├─ Mix of severity levels
└─ Realistic moderation scenarios

sample_classroom.csv:
├─ 40-50% concerning (expected)
├─ Educational context
└─ Peer feedback patterns
```

---

## 🆘 Need Help?

1. **Try sample files first** → Verify system works
2. **Read error messages** → Often point to solution
3. **Check file format** → Must be CSV/XLSX/XLS
4. **Review documentation** → Comprehensive guides available
5. **Test with small files** → 10-20 entries to start

---

## ✨ Pro Tips

💡 **Start small** - Test with 10-50 entries before bulk uploads  
💡 **Use filters** - Focus on concerning content to save time  
💡 **Export regularly** - Download results for record-keeping  
💡 **Track trends** - Compare analyses week-over-week  
💡 **Combine methods** - Use bulk for overview, single-text for deep dives  
💡 **Review context** - AI isn't perfect, use human judgment  

---

## 🎓 Remember

```
┌────────────────────────────────────────────────────┐
│  "AI assists, humans decide"                       │
│                                                     │
│  Use bulk analysis to:                             │
│  ✓ Identify patterns                               │
│  ✓ Prioritize review                               │
│  ✓ Track metrics                                   │
│  ✓ Save time                                       │
│                                                     │
│  But always:                                       │
│  ✓ Consider context                                │
│  ✓ Apply human judgment                            │
│  ✓ Be consistent and fair                          │
└────────────────────────────────────────────────────┘
```

---

**Ready to analyze? Upload your first CSV!** 🚀

_Last Updated: October 2024 • Version 1.0_

