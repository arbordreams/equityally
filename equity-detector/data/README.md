# Data Directory

This directory contains all data files used in EquityAlly, organized into two categories:

---

## Directory Structure

```
data/
├── samples/           # Sample CSV files for testing
└── training/          # Training datasets
```

---

## Samples (`samples/`)

Sample CSV files for testing the bulk analysis feature.

**Files:**
- `sample_data.csv` (20 entries) - General testing
- `sample_social_media.csv` (15 entries) - Social media context
- `sample_classroom.csv` (15 entries) - Educational context

**Purpose:**
- Demo and testing the bulk analysis feature
- User onboarding and tutorials
- Quick validation of CSV upload functionality

**Documentation:** See `samples/README.md`

---

## Training Data (`training/`)

Datasets used to train the BERT model.

**Main Dataset:**
- Jigsaw Toxic Comment Classification Challenge
  - 120,000+ labeled comments
  - 6 toxicity categories
  - Multi-label classification data

**Additional Sources** (not included in repo):
- Wikipedia Talk Pages
- Twitter toxicity datasets
- Civil Comments
- Q&A forum data

**Documentation:** See `training/README.md`

---

## Data Usage

### For End Users
Use the sample files in `samples/` to test bulk analysis:
1. Navigate to **🔍 Detector** page
2. Select **Bulk Analysis** tab
3. Upload a sample CSV file
4. View results and download analyzed data

### For Developers
Access training data in `training/` for:
- Model evaluation
- Performance benchmarking
- Additional fine-tuning
- Research and analysis

---

## CSV Format Requirements

All CSV files for bulk analysis must follow this format:

```csv
text
"First message here..."
"Second message here..."
...
```

**Requirements:**
- Must have a `text` column header
- One text entry per row
- UTF-8 encoding
- Text in quotes if containing commas

---

## File Size Considerations

### Sample Files
- Total size: ~4KB (tracked in git)
- Safe to commit and version

### Training Files
- Total size: ~500MB+ (may be gitignored)
- Download separately if needed
- See `training/README.md` for sources

---

## Related Documentation

### Sample Data
- `samples/README.md` - Sample file details
- `docs/SAMPLE_FILES_README.md` - Comprehensive sample guide
- `docs/BULK_ANALYSIS_GUIDE.md` - Bulk analysis tutorial

### Training Data
- `training/README.md` - Training data overview
- `docs/dataset_card.md` - Dataset documentation
- `docs/model_card.md` - Model documentation
- `evaluation/` - Performance metrics

---

**Last Updated**: October 23, 2024

