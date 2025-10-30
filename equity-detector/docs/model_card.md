# Model Card: BERT Toxicity Classifier

## Model Details

- **Model Name**: Equity Ally BERT Toxicity Classifier
- **Model Type**: BERT (Bidirectional Encoder Representations from Transformers)
- **Architecture**: BertForSequenceClassification
- **Base Model**: bert-base-uncased (Google)
- **Parameters**: ~110M trainable
- **Framework**: PyTorch + HuggingFace Transformers
- **Task**: Binary toxicity classification (any_toxic aggregated label)

## Development Pipeline

### Stage 1: Fine-Tuning

**Training Data:** 120,000+ carefully balanced samples from 5 datasets

**Datasets:**
1. **Jigsaw Toxic Comment Classification** (Kaggle)
   - 160,000+ Wikipedia talk page comments
   - 6 toxicity labels (toxic, severe_toxic, obscene, threat, insult, identity_hate)

2. **Civil Comments** (Jigsaw)
   - 2 million+ public comments
   - Multi-aspect toxicity + identity annotations

3. **Twitter Cyberbullying Dataset**
   - 47,000+ labeled tweets
   - Multi-class cyberbullying categories

4. **Hate Speech & Offensive Language** (Davidson et al.)
   - 25,000+ labeled tweets
   - Distinguishes hate speech from offensive language

5. **Formspring Cyberbullying Dataset**
   - 12,000+ Q&A platform posts
   - Binary cyberbullying labels

**Training Configuration:**
- Base: bert-base-uncased (pretrained on BookCorpus + Wikipedia)
- Learning Rate: 2e-5 (AdamW)
- Batch Size: 32
- Epochs: 3-4 (early stopping)
- Max Sequence Length: 256 tokens
- Warmup: 10% of steps
- Loss: Binary Cross Entropy
- Validation Split: 20% holdout
- Hardware: GPU-accelerated

**Balanced Sampling:**
- Equal representation across toxicity types
- Diverse platform coverage
- Demographic fairness
- Prevents overfitting to single source

### Stage 2: Post-Hoc Calibration

**Method:** Isotonic Regression (recommended) / Temperature Scaling (alternative)

**Purpose:** Ensures predicted probabilities align with true frequencies

**Result:** Validation accuracy improved from 89.94% to 96.84%

## Performance

### Recommended Configuration

- **Calibration Method**: **Isotonic Regression**
- **Alternative**: Temperature Scaling (96.77% accuracy, simpler)
- **Decision Threshold**: 0.40 (F1-optimal on validation, calibrated)

### Metrics (Isotonic Calibration)

**Validation Set (10,000 samples):**
- **Accuracy: 96.84%** ⭐
- **F1 Score: 84.87%**
- Precision: 81.75% | Recall: 88.26%
- ROC-AUC: 96.80% | PR-AUC: 92.43%
- False Positive Rate: 2.20% | False Negative Rate: 11.74%

**Test Set (5,000 samples):**
- **Accuracy: 91.52%**
- **F1 Score: 67.19%**
- Precision: 54.98% | Recall: 86.68%
- ROC-AUC: 96.80% | PR-AUC: 79.34%
- False Positive Rate: 7.94% | False Negative Rate: 13.32%

### Calibration Comparison

| Method | Val Accuracy | Test Accuracy | Val F1 | Test F1 |
|--------|--------------|---------------|--------|---------|
| Uncalibrated | 89.94% | 89.94% | 74.35% | 58.09% |
| Temperature | 96.77% | 91.20% | 84.73% | 66.84% |
| **Isotonic** ✅ | **96.84%** | **91.52%** | **84.87%** | **67.19%** |
| Platt | 96.80% | 91.38% | 84.85% | 67.14% |

## Intended Use

**Designed for:**
- Content moderation systems
- Community management tools
- Educational platforms
- Social media safety
- Research on online toxicity

**Out-of-Scope:**
- Automated content removal without human review
- Legal decision-making
- High-stakes decisions without human oversight
- Non-English text (model optimized for English only)

## Limitations

1. **Language**: English only (bert-base-uncased is English-focused)
2. **Context Window**: Limited to first 256 tokens
3. **Dataset Bias**: May reflect biases in training data (Wikipedia, Twitter)
4. **Domain Shift**: Performance may vary on platforms not in training data
5. **Evolving Language**: May not capture newest slang or coded toxicity
6. **Aggregated Labels**: Uses "any_toxic" - doesn't distinguish toxicity types

## Ethical Considerations

- **False Positives**: May flag legitimate discourse (journalism, education, quotes)
- **False Negatives**: May miss subtle, coded, or context-dependent toxicity
- **Human Review**: Predictions should inform, not replace, human judgment
- **Fairness**: Regular auditing recommended for demographic bias
- **Transparency**: Open-source model and documentation
- **Privacy**: Designed for on-device inference (no data transmission)

## References

**Architecture:**
- Devlin et al. (2018): [BERT](https://arxiv.org/abs/1810.04805)

**Calibration:**
- Guo et al. (2017): [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599)

**Datasets:**
- [Jigsaw Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
- [Civil Comments](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification)
- [Twitter Cyberbullying](https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification)
- [Hate Speech & Offensive Language](https://github.com/t-davidson/hate-speech-and-offensive-language)
- [Formspring Cyberbullying](https://www.kaggle.com/datasets/swetaagrawal/formspring-data-for-cyberbullying-detection)
