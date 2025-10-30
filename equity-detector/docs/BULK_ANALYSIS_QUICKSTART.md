# Bulk Analysis Quick Start ⚡

## 3 Simple Steps

### 1️⃣ Prepare Your File
```csv
id,text,author
1,"Great work!",User1
2,"You're terrible",User2
3,"Thanks for sharing",User3
```
✅ CSV, XLSX, or XLS format  
✅ One column with text  
✅ Optional ID column for tracking

---

### 2️⃣ Upload & Configure
1. Navigate to **Detector** page
2. Select **"📊 Upload CSV/Bulk Analysis"**
3. Upload your file
4. Choose text column
5. Choose ID column (optional)

---

### 3️⃣ Analyze & Download
Click **"🔍 Analyze All Entries"**

**You get:**
- ✅ Summary metrics
- 📊 4 interactive charts
- 📋 Filterable results table
- 💾 Downloadable CSV & summary
- 💡 AI-powered insights

---

## What You'll See

### 📊 Summary Metrics
- Total entries analyzed
- Safe vs. Concerning counts
- Average risk score
- Processing time

### 📈 Visualizations

#### 1. **Distribution Pie Chart**
Safe vs. Concerning percentage breakdown

#### 2. **Risk Score Histogram**
How risk scores are distributed across entries

#### 3. **Box Plot**
Statistical comparison between safe/concerning content

#### 4. **Severity Breakdown**
Low/Moderate/High concern categories

### 📋 Interactive Table
- Filter: All / Safe / Concerning / High Risk
- Sort: By risk score or original order
- Shows: Text preview, probabilities, classification

### 💾 Downloads

**CSV Results:**
- All entries with full text
- Risk scores and classifications
- Uncertainty metrics (if MC Dropout enabled)

**Summary Report:**
- Total counts and percentages
- Statistical measures
- Analysis timestamp
- Configuration details

---

## Tips for Best Results

### 📝 Data Preparation
- ✅ Clean data (remove duplicates)
- ✅ Text length > 5 characters
- ✅ UTF-8 encoding
- ✅ Test with 10-50 entries first

### ⚙️ Advanced Options
- Enable **Monte Carlo Dropout** for uncertainty metrics
- Use 20-30 passes for balance of speed/accuracy
- Good for critical applications

### 🎯 Filtering Results
- **"High Risk Only"** - Focus on entries >70% risk
- **"Concerning Only"** - All flagged content
- **Sort by Risk Score** - Prioritize review

### 📊 Interpreting Scores
- **0-50%**: Low concern - generally safe
- **30-70%**: Moderate - monitor and contextual review
- **80-90%**: High risk - review recommended
- **90-100%**: Critical - immediate attention

---

## Common Use Cases

### 🏫 Schools
"Process 500 student forum posts overnight"
- Upload posts CSV
- Review flagged content in morning
- Download report for records

### 👥 Communities
"Moderate weekly comment batches"
- Export week's comments
- Analyze in bulk
- Focus moderation on high-risk

### 🔬 Research
"Analyze dataset trends"
- Upload research data
- Get distribution statistics
- Export results with uncertainty

---

## Performance Guide

### Processing Speed
| Entries | Standard Mode | MC Dropout (20x) |
|---------|---------------|------------------|
| 10      | ~1-3 sec      | ~20-60 sec      |
| 100     | ~10-30 sec    | ~3-10 min       |
| 1000    | ~2-5 min      | ~30-100 min     |

💡 **Tip:** For large datasets, start without MC Dropout, then re-analyze high-risk entries with MC enabled.

---

## Troubleshooting

### ❌ "No valid entries found"
- Check text column has content
- Ensure entries are >5 characters
- Verify correct column selected

### ⏱️ "Taking too long"
- Expected for large files
- Disable MC Dropout for speed
- Split into smaller batches

### 📁 "File upload failed"
- Verify CSV/Excel format
- Check file isn't corrupted
- Try saving as plain CSV

---

## Example Workflow

```mermaid
graph LR
    A[Export Comments] --> B[Upload CSV]
    B --> C[Select Columns]
    C --> D[Analyze]
    D --> E[Review Charts]
    E --> F[Filter High Risk]
    F --> G[Download Results]
    G --> H[Manual Review]
```

**Time Investment:**
- Setup: 30 seconds
- Analysis: 1-5 minutes (100 entries)
- Review: 5-15 minutes
- **Total: ~10-20 minutes for 100 entries**

---

## Sample Data Included

Use `sample_data.csv` (20 entries) to test:
```bash
# In Streamlit app:
1. Select "📊 Upload CSV/Bulk Analysis"
2. Upload sample_data.csv
3. Select "text" column
4. Select "id" column (optional)
5. Click "🔍 Analyze All Entries"
```

Expected results:
- ~50% concerning content
- Mix of low, moderate, and high risk
- Good for testing filtering/sorting

---

## Need More Help?

📖 **Full Guide:** See [BULK_ANALYSIS_GUIDE.md](BULK_ANALYSIS_GUIDE.md)  
🏠 **Main Docs:** See [README.md](README.md)  
🔧 **Technical:** Check [utils/bert_model.py](utils/bert_model.py)

---

## Quick Reference

| Feature | Location | Purpose |
|---------|----------|---------|
| Upload | Step 1 | Select CSV/Excel file |
| Column Selection | Step 1 | Choose text & ID columns |
| Advanced Options | Expander | Enable MC Dropout |
| Analyze Button | Step 2 | Start processing |
| Visualizations | Step 3 | See distribution charts |
| Filters | Results Table | Focus on specific content |
| Downloads | Results Section | Export CSV & summary |
| Insights | Bottom | AI-generated recommendations |

---

**Ready to analyze? Upload your first CSV file!** 🚀

For detailed documentation, see the complete [Bulk Analysis Guide](BULK_ANALYSIS_GUIDE.md).

