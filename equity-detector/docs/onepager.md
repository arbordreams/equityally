# Executive Summary: BERT Toxicity Model

## Model Development Pipeline

### Stage 1: Multi-Dataset Fine-Tuning

**Base Model:** bert-base-uncased (Google, 110M parameters)

**Training Data:** 120,000+ carefully balanced samples from:
- Jigsaw Toxic Comments (Wikipedia, 160K+ comments)
- Civil Comments (2M+ public comments)
- Twitter Cyberbullying (47K+ tweets)
- Hate Speech & Offensive Language (25K+ tweets)
- Formspring Cyberbullying (12K+ Q&A posts)

**Training Configuration:**
- Learning Rate: 2e-5 (AdamW)
- Batch Size: 32
- Epochs: 3-4 (early stopping)
- Max Sequence: 256 tokens
- Loss: Binary Cross Entropy
- Validation: 20% holdout

### Stage 2: Post-Hoc Calibration

**Method:** Isotonic Regression (recommended)

**Improvement:** Validation accuracy 89.94% → **96.84%** (+6.9 pp)

## Performance Metrics

**Validation Set (10,000 samples):**
- **Accuracy: 96.84%** ⭐
- **F1 Score: 84.87%**
- Precision: 81.75% | Recall: 88.26%
- ROC-AUC: 96.80% | PR-AUC: 92.43%

**Test Set (5,000 samples):**
- **Accuracy: 91.52%**
- **F1 Score: 67.19%**
- Precision: 54.98% | Recall: 86.68%
- ROC-AUC: 96.80% | PR-AUC: 79.34%

### Optimal Decision Threshold

| Label | Threshold | Method |
|-------|-----------|--------|
| any_toxic | **0.40** | F1-optimal (Isotonic) |

## Calibration Impact

Isotonic regression calibration provides:
- **+6.9 pp accuracy** improvement on validation (89.94% → 96.84%)
- **+10.5 pp F1 improvement** on validation (74.35% → 84.87%)
- **Reliable probability estimates** for confident decision-making
- **Minimal overhead** (<1ms per prediction)

**Alternative:** Temperature Scaling achieves 96.77% validation accuracy with simpler implementation.

## Deployment Recommendations

1. **Use Isotonic calibration** for best accuracy and F1 scores
2. **Apply threshold = 0.40** for binary classification
3. **Monitor calibration** on production data; recalibrate if distribution shifts
4. **Human review** recommended for borderline cases (probabilities 0.70-0.85)
5. **Consider Temperature Scaling** if implementation simplicity is critical

## Model Specifications

- **Architecture:** BERT-base-uncased (110M parameters)
- **Training:** 120,000+ samples from 5 datasets
- **Calibration Method:** Isotonic Regression (primary) / Temperature Scaling (alternative)
- **Inference Latency:** ~100ms on CPU (calibration adds <1ms)
- **Memory:** ~418MB model + minimal calibration parameters
- **Throughput:** 600+ predictions/minute (batch_size=32)
- **Privacy:** 100% on-device, offline capable

## Documentation

For detailed analysis:
- `docs/calibration.md` - Comprehensive calibration comparison
- `docs/model_card.md` - Full model documentation
- `visualizations/calibrated/` - All performance plots
- `evaluation/` - Raw metrics and parameters
