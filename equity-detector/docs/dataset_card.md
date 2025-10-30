# Dataset Card

## Overview

This dataset contains toxic comment data from the Jigsaw Toxic Comment Classification Challenge.

## Statistics

### Dataset Splits

| Split      | Samples    |
|------------|-----------|
| Validation | 159,571 |
| Test       | 63,978 |

### Label Distribution

#### Validation Set

| Label          | Count     | Percentage |
|----------------|-----------|------------|
| toxic          |    15,294 |      9.58% |
| severe_toxic   |     1,595 |      1.00% |
| obscene        |     8,449 |      5.29% |
| threat         |       478 |      0.30% |
| insult         |     7,877 |      4.94% |
| identity_hate  |     1,405 |      0.88% |

#### Test Set

| Label          | Count     | Percentage |
|----------------|-----------|------------|
| toxic          |     6,090 |      9.52% |
| severe_toxic   |       367 |      0.57% |
| obscene        |     3,691 |      5.77% |
| threat         |       211 |      0.33% |
| insult         |     3,427 |      5.36% |
| identity_hate  |       712 |      1.11% |

## Schema

- **Text Column**: `comment_text`
- **Label Columns**: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`
- **Label Type**: Multi-label binary (0 or 1 for each label)

## Class Imbalance

The dataset exhibits significant class imbalance, with most comments being non-toxic. See `visualizations/class_imbalance_heatmap.png` for label co-occurrence patterns.
