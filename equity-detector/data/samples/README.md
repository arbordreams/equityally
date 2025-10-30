# Sample Data Files

This directory contains sample CSV files for testing the EquityAlly bulk analysis feature.

## Files

### `sample_data.csv` (20 entries)
General testing dataset with a mix of safe and concerning content.
- **Purpose**: General feature testing
- **Entries**: 20
- **Distribution**: ~50% safe, ~50% concerning
- **Context**: Mixed (social media, comments, messages)

**Use case**: First-time users learning the bulk analysis feature

---

### `sample_social_media.csv` (15 entries)
Social media and forum posts with varying toxicity levels.
- **Purpose**: Social platform content moderation
- **Entries**: 15
- **Distribution**: ~45-50% concerning
- **Context**: Social media posts, forum comments

**Use case**: Testing content moderation for social platforms

---

### `sample_classroom.csv` (15 entries)
Student comments and classroom discussion examples.
- **Purpose**: Educational environment content monitoring
- **Entries**: 15
- **Distribution**: ~45-50% concerning
- **Context**: Student forum posts, assignment comments

**Use case**: Schools and educators monitoring student interactions

---

## Usage

### Basic Usage
1. Navigate to the **🔍 Detector** page in EquityAlly
2. Switch to the **"Bulk Analysis"** tab
3. Upload one of these sample files
4. Click **"Analyze CSV"**
5. View results with visualizations and download the analyzed CSV

### Expected Results

| File | Total Entries | ~Safe | ~Concerning | Concerning % |
|------|---------------|-------|-------------|--------------|
| sample_data.csv | 20 | ~10 | ~10 | ~45-55% |
| sample_social_media.csv | 15 | ~8 | ~7 | ~45-50% |
| sample_classroom.csv | 15 | ~8 | ~7 | ~45-50% |

*Note: Exact counts may vary slightly based on model version and calibration settings.*

---

## CSV Format

All sample files follow this structure:

```csv
text
"First message here..."
"Second message here..."
...
```

**Requirements:**
- Must have a `text` column header
- One text entry per row
- Text should be in quotes if it contains commas
- UTF-8 encoding

---

## Creating Your Own CSV

To create your own bulk analysis file:

1. Create a CSV file with a `text` column header
2. Add one text entry per row
3. Save as UTF-8 encoding
4. Upload to EquityAlly's Bulk Analysis feature

**Example:**
```csv
text
"This is a great comment!"
"You're terrible at this and should quit"
"I love learning new things"
"What an idiot"
```

---

## Related Documentation

- **Quick Start**: `equity-detector/docs/BULK_ANALYSIS_QUICKSTART.md`
- **Full Guide**: `equity-detector/docs/BULK_ANALYSIS_GUIDE.md`
- **Cheat Sheet**: `equity-detector/docs/BULK_ANALYSIS_CHEATSHEET.md`
- **Detailed Sample Info**: `equity-detector/docs/SAMPLE_FILES_README.md`

---

**Last Updated**: October 23, 2024

