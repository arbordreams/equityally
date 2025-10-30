# Sample CSV Files for Bulk Analysis Testing

This directory includes several sample CSV files to help you test and learn the bulk analysis feature.

## Available Sample Files

### 1. `sample_data.csv` (20 entries)
**Purpose:** General testing and demonstration

**Columns:**
- `id` - Unique identifier
- `text` - Content to analyze
- `author` - Username

**Content Mix:**
- 50% safe content
- 50% concerning content
- Mix of severity levels

**Best For:**
- First-time users learning the system
- Testing all visualization features
- Understanding classification outputs

---

### 2. `sample_social_media.csv` (15 entries)
**Purpose:** Social media / forum moderation

**Columns:**
- `post_id` - Post identifier
- `username` - User who posted
- `comment` - Comment text
- `timestamp` - When posted
- `platform` - Source platform

**Content Mix:**
- Typical forum/social media comments
- Mix of supportive and toxic content
- Includes timestamps for trend analysis

**Best For:**
- Social media managers
- Community moderators
- Testing with realistic forum data

---

### 3. `sample_classroom.csv` (15 entries)
**Purpose:** Educational / school environment

**Columns:**
- `student_id` - Anonymized student ID
- `message` - Student message/comment
- `assignment` - Context (discussion, peer review, etc.)
- `date` - Submission date

**Content Mix:**
- Peer feedback and discussions
- Mix of constructive and inappropriate comments
- Educational context examples

**Best For:**
- Schools and teachers
- Educational technology platforms
- Testing in academic settings

---

## How to Use

### Quick Test
1. Open EquityLens Detector page
2. Select "📊 Upload CSV/Bulk Analysis"
3. Upload one of the sample files
4. Select the `text`, `comment`, or `message` column
5. Optionally select an ID column
6. Click "Analyze All Entries"

### What to Look For

#### With `sample_data.csv`:
- **Expected:** ~50% concerning, 50% safe
- **Test:** All visualization types
- **Practice:** Filtering and sorting

#### With `sample_social_media.csv`:
- **Expected:** ~40-50% concerning
- **Test:** ID tracking with usernames
- **Practice:** Identifying problematic users

#### With `sample_classroom.csv`:
- **Expected:** ~40-50% concerning
- **Test:** Context-based analysis
- **Practice:** Educational moderation workflows

---

## Creating Your Own Sample Files

### Minimum Requirements
```csv
text
"Your content here"
"Another entry"
```

### Recommended Format
```csv
id,text,optional_context
1,"Content to analyze","Additional info"
2,"More content","More info"
```

### Best Practices
- ✅ Use UTF-8 encoding
- ✅ Quote text fields with commas
- ✅ Include headers in first row
- ✅ Keep text entries >5 characters
- ✅ Use unique IDs for tracking

### File Formats Supported
- `.csv` - Comma-separated values (recommended)
- `.xlsx` - Excel 2007+ (requires openpyxl)
- `.xls` - Excel legacy (requires openpyxl)

---

## Expected Results Summary

| File | Total | Safe | Concerning | Avg Risk |
|------|-------|------|------------|----------|
| sample_data.csv | 20 | ~10 | ~10 | ~45-55% |
| sample_social_media.csv | 15 | ~8 | ~7 | ~45-50% |
| sample_classroom.csv | 15 | ~8 | ~7 | ~45-50% |

*Note: Exact results may vary slightly based on model version and settings*

---

## Testing Workflows

### Basic Test (5 minutes)
1. Upload `sample_data.csv`
2. Review all 4 visualizations
3. Test filtering options
4. Download results CSV
5. Review summary report

### Advanced Test (10 minutes)
1. Upload `sample_social_media.csv`
2. Enable Monte Carlo Dropout (20 passes)
3. Analyze with uncertainty metrics
4. Filter for "High Risk Only"
5. Sort by risk score
6. Download and compare results

### Batch Comparison Test (15 minutes)
1. Analyze `sample_data.csv` - save results
2. Analyze `sample_social_media.csv` - save results
3. Analyze `sample_classroom.csv` - save results
4. Compare distributions across files
5. Identify patterns and differences

---

## Privacy & Data Handling

### Sample Files
- ✅ All data is synthetic/fictional
- ✅ No real users or personal information
- ✅ Safe for testing and demonstrations
- ✅ Can be shared publicly

### Your Own Data
- 🔒 All analysis runs locally (BERT)
- 🔒 Data never leaves your device
- 🔒 Results only saved if you download
- 🔒 No cloud storage or external APIs
- ⚠️ OpenAI features require API (optional)

---

## Troubleshooting Sample Files

### "No valid entries found"
- Make sure you selected the correct text column
- Verify the file uploaded successfully
- Check for file corruption

### "File upload failed"
- Try opening in Excel and re-saving
- Verify file format (.csv, .xlsx, .xls)
- Check file isn't password-protected

### Unexpected Results
- Results may vary by ~5% due to model behavior
- Monte Carlo Dropout adds randomness (expected)
- Context affects interpretation (normal)

---

## Creating Domain-Specific Test Files

### For Social Media
Include columns:
- Username, timestamp, platform, reactions, replies
- Mix casual chat, heated discussions, toxic comments

### For Education
Include columns:
- Student ID, assignment type, submission date
- Mix peer feedback, discussions, reflections

### For Community Forums
Include columns:
- Thread ID, post position, user reputation
- Mix questions, answers, off-topic, conflicts

### For Customer Service
Include columns:
- Ticket ID, customer ID, response time
- Mix complaints, compliments, frustrations

---

## Need More Help?

📖 **Full Documentation:**
- [Bulk Analysis Guide](BULK_ANALYSIS_GUIDE.md)
- [Quick Start](BULK_ANALYSIS_QUICKSTART.md)
- [Main README](README.md)

🔧 **Technical Support:**
- Check file format requirements
- Verify column types
- Review error messages

💡 **Tips:**
- Start with small test files (10-20 entries)
- Validate results make sense
- Gradually increase to larger datasets
- Use filters to focus on concerning content

---

**Ready to create your own CSV files for analysis?**

Remember: These samples are here to help you learn. Once comfortable, upload your own data for real-world content moderation!

