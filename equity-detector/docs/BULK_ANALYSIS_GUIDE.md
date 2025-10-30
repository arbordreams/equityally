# Bulk Analysis Guide - EquityLens

## Overview
The Bulk Analysis feature allows you to analyze multiple text entries at once from CSV or Excel files, providing comprehensive statistics, visualizations, and downloadable results.

## Getting Started

### 1. Prepare Your Data File

Your CSV or Excel file should contain:
- **Required**: A column with text content to analyze
- **Optional**: An ID or name column for tracking individual entries

#### Supported File Formats
- `.csv` - Comma-separated values
- `.xlsx` - Excel workbook (2007+)
- `.xls` - Excel workbook (legacy)

#### Example CSV Structure
```csv
id,text,author
1,"Great job on the presentation!",User123
2,"You're such an idiot.",User456
3,"Thanks for the helpful info.",User789
```

### 2. Upload and Configure

1. Navigate to the Detector page
2. Select "📊 Upload CSV/Bulk Analysis" as your input method
3. Upload your file using the file uploader
4. Preview the first 5 rows to verify the upload
5. Select the column containing text to analyze
6. Optionally select an ID/Name column for tracking

### 3. Run Analysis

Click "🔍 Analyze All Entries" to start the batch analysis. The system will:
- Filter out empty or very short texts (< 5 characters)
- Show a progress bar during analysis
- Process each entry through the BERT model
- Calculate risk scores and classifications

## Features

### 📊 Analysis Summary
Get instant metrics including:
- Total entries analyzed
- Number of safe vs. concerning content
- Average risk score
- Processing time and speed

### 📈 Visualizations

#### 1. Classification Distribution (Pie Chart)
- Visual breakdown of safe vs. concerning content
- Percentage and count for each category

#### 2. Risk Score Distribution (Histogram)
- Shows the distribution of risk scores across all entries
- Highlights the 40% decision threshold (F1-optimal)
- Color-coded by risk level

#### 3. Risk Score by Classification (Box Plot)
- Statistical comparison between safe and concerning content
- Shows median, quartiles, and outliers
- Includes mean and standard deviation

#### 4. Severity Level Breakdown (Bar Chart)
- Categorizes content into:
  - **Low Concern** (< 30%)
  - **Moderate Concern** (30-70%)
  - **High Concern** (> 70%)

### 📋 Detailed Results Table

Interactive table with:
- **Filtering**: View all, safe only, concerning only, or high risk only
- **Sorting**: By original order or risk score (high to low / low to high)
- **Columns**:
  - Text preview (truncated to 100 characters)
  - Safe probability percentage
  - Concerning probability percentage
  - Classification (Safe/Concerning)
  - Uncertainty percentage (if Monte Carlo Dropout is enabled)

### 💾 Download Results

Two download options:

#### 1. CSV Results
Complete analysis results including:
- Full text (not truncated)
- All probability scores
- Classifications
- Original indices
- ID column (if provided)

Filename format: `equity_analysis_results_YYYYMMDD_HHMMSS.csv`

#### 2. Summary Report
Text file with:
- Total entries and counts
- Percentages for each category
- Statistical measures (min, max, average)
- Analysis timestamp
- Processing time
- Configuration details

Filename format: `equity_analysis_summary_YYYYMMDD_HHMMSS.txt`

### 💡 Key Insights

Automated insights generation including:
- Overall safety assessment
- High-risk entry count and recommendations
- Moderate-risk entry identification
- Uncertainty warnings (when using Monte Carlo Dropout)

#### Alert Levels
- **✅ Overall Positive**: < 25% concerning content
- **⚠️ Moderate Concern**: 25-50% concerning content
- **🚨 High Alert**: > 50% concerning content

## Advanced Options

### Monte Carlo Dropout for Bulk Analysis

Enable Monte Carlo Dropout in the Advanced Options section before running analysis:

**Benefits:**
- More robust predictions through ensemble averaging
- Uncertainty quantification for each entry
- Better identification of borderline cases

**Considerations:**
- Increases processing time (multiply by number of passes)
- Recommended for critical applications
- 20-30 passes recommended for balance

**Example Processing Times:**
- Standard: ~0.1-0.3s per entry
- MC Dropout (20 passes): ~2-6s per entry

## Use Cases

### 1. Social Media Monitoring
- Analyze user comments for toxicity
- Identify concerning patterns
- Prioritize moderation efforts

### 2. Community Management
- Review forum posts in bulk
- Track community health metrics
- Generate periodic safety reports

### 3. Content Moderation Queue
- Process moderation queue efficiently
- Flag high-risk content automatically
- Focus human review on borderline cases

### 4. Research and Analysis
- Study content patterns over time
- Compare different communities or platforms
- Validate moderation strategies

### 5. Compliance Reporting
- Generate safety compliance reports
- Document moderation efforts
- Track improvement metrics

## Best Practices

### Data Preparation
1. **Clean Your Data**: Remove duplicate entries, empty cells
2. **Text Length**: Ensure entries have sufficient content (> 5 characters)
3. **Encoding**: Use UTF-8 encoding for special characters
4. **File Size**: For large datasets (> 1000 entries), consider splitting into batches

### Analysis Strategy
1. **Start Small**: Test with 10-50 entries first
2. **Review Threshold**: Use filters to focus on concerning content
3. **Context Matters**: Remember AI isn't perfect - review high-risk items manually
4. **Track Trends**: Use regular bulk analysis to monitor changes over time

### Interpreting Results

#### Risk Score Interpretation
- **0-50%**: Low risk - Generally safe content
- **30-70%**: Moderate - Worth monitoring and contextual review
- **80-90%**: High risk - Review recommended
- **90-100%**: Very high risk - Immediate review needed

#### When to Use MC Dropout
- Critical applications requiring high confidence
- Borderline cases need careful evaluation
- Research requiring uncertainty metrics
- When false positives are costly

## Troubleshooting

### Common Issues

**File Upload Fails**
- Ensure file is valid CSV or Excel format
- Check file isn't corrupted or password-protected
- Try saving as CSV if using Excel

**No Valid Entries Found**
- Check text column has actual content
- Ensure text length > 5 characters
- Verify column selection is correct

**Slow Processing**
- Expected for large datasets
- Consider disabling MC Dropout for speed
- Split large files into smaller batches

**Unexpected Classifications**
- Remember: context matters
- AI can misinterpret sarcasm, quotes
- Always review high-stakes decisions manually
- Consider cultural and linguistic nuances

## Sample Data

A sample CSV file (`sample_data.csv`) is included with 20 entries mixing safe and concerning content. Use this to test the bulk analysis feature.

## Privacy & Security

- All analysis runs locally on your device
- No data is sent to external servers (BERT model)
- OpenAI API only used if you enable AI Assistant features
- Downloaded results stay on your local machine

## Technical Details

### Performance
- **Speed**: BERT-base model: ~100-300ms per entry (CPU), faster on GPU
- **ROC-AUC**: 96.8% (excellent discrimination)
- **F1 Score**: 58.1% (balanced precision-recall)
- **Linear scaling**: Performance scales linearly with number of entries
- **Monte Carlo Dropout**: Uncertainty quantification + ensemble averaging
- **AI Verification**: Enhanced contextual analysis

### Model Details
- Architecture: BERT-base (110M parameters, 418MB)
- Training: 120,000+ samples from 5 datasets
- Calibration: Isotonic Regression (96.8% validation accuracy)
- Decision threshold: 40% (F1-optimized)
- Classes: Safe (0) / Concerning (1)

## Support

For issues, questions, or feature requests:
1. Check this guide first
2. Review the main README.md
3. Ensure all dependencies are installed
4. Verify model files are properly loaded

## Future Enhancements

Planned features:
- Export to Excel with formatting
- Trend analysis over time
- Comparison between multiple files
- Custom threshold configuration
- Batch AI verification
- More visualization options

---

**Version**: 1.0  
**Last Updated**: October 2025  
**Compatible with**: EquityLens v1.0+

