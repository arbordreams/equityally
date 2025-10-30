# Training Data

This directory contains the datasets used to train the EquityAlly BERT model.

## Jigsaw Toxic Comment Classification Challenge

The primary training dataset from Kaggle's Jigsaw Toxic Comment Classification Challenge.

### Dataset Overview
- **Source**: [Kaggle Competition](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
- **Size**: 120,000+ labeled comments
- **Labels**: 6 toxicity categories (toxic, severe_toxic, obscene, threat, insult, identity_hate)
- **License**: CC0 (Public Domain)

### Files

```
jigsaw-toxic-comments/
├── train.csv          # Training data (~160K comments)
├── test.csv           # Test data (~150K comments)
├── test_labels.csv    # Test labels
└── *.csv.zip          # Compressed versions
```

### Data Format

**train.csv** columns:
- `id`: Unique identifier
- `comment_text`: The text content
- `toxic`: Binary label (1 = present, 0 = absent) — any toxic content
- `severe_toxic`: Binary label (1 = present, 0 = absent) — highly toxic content
- `obscene`: Binary label (1 = present, 0 = absent) — profanity/sexual explicitness
- `threat`: Binary label (1 = present, 0 = absent) — threats of harm or violence
- `insult`: Binary label (1 = present, 0 = absent) — insulting/abusive language
- `identity_hate`: Binary label (1 = present, 0 = absent) — hate speech targeting identity

### Training Approach

EquityAlly was trained using a **multi-dataset approach** combining:

1. **Jigsaw Toxic Comments** (this dataset)
2. Wikipedia Talk Pages
3. Twitter toxicity datasets
4. Civil Comments
5. Online Q&A forum data

Total: **120,000+ samples** across **5 distinct datasets**

### Label Aggregation

For binary classification (safe vs. concerning), we use:
```python
# A comment is "concerning" if ANY toxicity label is 1
is_concerning = (toxic OR severe_toxic OR obscene OR threat OR insult OR identity_hate)
```

### Model Training Pipeline

1. **Data Preprocessing**
   - Text cleaning and normalization
   - Multi-dataset aggregation
   - Binary label creation

2. **Fine-tuning**
   - Base model: `bert-base-uncased` (110M parameters)
   - Training samples: 120,000+
   - Validation split: 80/20

3. **Calibration**
   - Method: Isotonic regression
   - Improvement: +6.9pp accuracy (89.94% → 96.84%)

### Performance on This Dataset

| Metric | Uncalibrated | Isotonic Calibrated |
|--------|--------------|---------------------|
| Accuracy | 89.94% | 96.84% |
| ROC-AUC | 95.2% | 96.8% |
| F1 Score | 82.1% | 84.9% |

---

## Additional Training Data Sources

While this directory contains only the Jigsaw dataset, the model was trained on multiple sources:

### Wikipedia Talk Pages
- **Size**: 20,000+ comments
- **Source**: Wikipedia toxic comment corpus
- **Focus**: Discussion page toxicity

### Twitter Dataset
- **Size**: 15,000+ tweets
- **Source**: Public Twitter toxicity datasets
- **Focus**: Short-form social media content

### Civil Comments
- **Size**: 10,000+ comments
- **Source**: Civil Comments dataset
- **Focus**: News article comments

### Q&A Forums
- **Size**: 5,000+ posts
- **Source**: Stack Exchange, Reddit moderation datasets
- **Focus**: Technical forum interactions

---

## Using This Data

### For Training
```python
import pandas as pd

# Load training data
df = pd.read_csv('jigsaw-toxic-comments/train.csv')

# Create binary label
df['is_concerning'] = (
    (df['toxic'] == 1) | 
    (df['severe_toxic'] == 1) | 
    (df['obscene'] == 1) | 
    (df['threat'] == 1) | 
    (df['insult'] == 1) | 
    (df['identity_hate'] == 1)
).astype(int)
```

### For Evaluation
The test set with labels is available for benchmarking:
```python
test_df = pd.read_csv('jigsaw-toxic-comments/test.csv')
test_labels = pd.read_csv('jigsaw-toxic-comments/test_labels.csv')
```

---

## Dataset Statistics

### Train Set
- Total comments: ~160,000
- Toxic comments: ~16,000 (~10%)
- Clean comments: ~144,000 (~90%)
- Average length: 394 characters

### Test Set
- Total comments: ~150,000
- Labeled subset: ~60,000
- Unlabeled subset: ~90,000

### Class Distribution
The dataset exhibits **class imbalance**:
- Safe: ~90%
- Concerning: ~10%

This is why calibration was critical for production deployment.

---

## Citation

If you use this dataset, please cite:

```
@misc{jigsaw-toxic-comment-classification-challenge,
    title = {Toxic Comment Classification Challenge},
    author = {Conversation AI team, a research initiative founded by Jigsaw and Google},
    year = {2018},
    howpublished = {\url{https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge}},
}
```

---

## License

- **Jigsaw Dataset**: CC0 (Public Domain)
- **EquityAlly Model**: MIT License (see repository root)

---

## Related Documentation

- **Model Card**: `equity-detector/docs/model_card.md`
- **Dataset Card**: `equity-detector/docs/dataset_card.md`
- **Calibration**: `equity-detector/docs/calibration.md`
- **Evaluation**: `equity-detector/evaluation/metrics_summary.csv`

---

**Note**: The training data files are large (>500MB total) and may be excluded from git via `.gitignore`. Download from the Kaggle competition page if needed.

**Last Updated**: October 23, 2024

