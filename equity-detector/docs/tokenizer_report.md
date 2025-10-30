# Tokenizer Analysis Report

## Overview

This report analyzes the BERT tokenizer's behavior on the toxicity dataset, including:
- Token length distributions
- Vocabulary coverage
- Rare and unknown tokens
- Toxic vs non-toxic token patterns

## Token Length Statistics

### Validation Set

- **Total tokens**: 2,000
- **Avg tokens/text**: 91.3
- **Median tokens/text**: 50.0
- **Max tokens/text**: 1987
- **Texts > MAX_LEN (256)**: 124 (6.2%)
- **Total [UNK] tokens**: 41

### Test Set

- **Total tokens**: 1,000
- **Avg tokens/text**: 85.2
- **Median tokens/text**: 49.0
- **Max tokens/text**: 1093
- **Texts > MAX_LEN (256)**: 52 (5.2%)
- **Total [UNK] tokens**: 44

## Vocabulary Analysis

### Top 20 Most Frequent Tokens (Validation)

| Rank | Token | Frequency |
|------|-------|----------|
|    1 | `.` | 8,162 |
|    2 | `the` | 5,923 |
|    3 | `,` | 5,758 |
|    4 | `"` | 4,961 |
|    5 | `to` | 3,563 |
|    6 | `i` | 3,175 |
|    7 | `you` | 3,028 |
|    8 | `of` | 2,806 |
|    9 | `'` | 2,704 |
|   10 | `a` | 2,701 |
|   11 | `and` | 2,665 |
|   12 | `is` | 2,219 |
|   13 | `that` | 2,009 |
|   14 | `it` | 1,828 |
|   15 | `in` | 1,774 |
|   16 | `!` | 1,519 |
|   17 | `this` | 1,228 |
|   18 | `for` | 1,218 |
|   19 | `-` | 1,173 |
|   20 | `not` | 1,152 |

## Key Findings

1. **Truncation Impact**: 6.2% of validation texts and 5.2% of test texts exceed MAX_LEN=256 and are truncated.

2. **Vocabulary Coverage**: The tokenizer has good coverage with minimal [UNK] tokens, indicating the domain is well-represented in BERT's vocabulary.

3. **Token Distribution**: Follows Zipf's law (see `visualizations/token_zipf_*.png`), with a small number of tokens accounting for most occurrences.

## Recommendations

- Consider increasing MAX_LEN to capture more context (current: 256, 6.2% truncated)
- Review rare toxic tokens for potential domain-specific vocabulary expansion
- Monitor fragmentation of toxic phrases to ensure critical context is preserved

## Visualizations

- Token length histograms: `visualizations/token_length_hist_*.png`
- Fragmentation box plots: `visualizations/token_fragmentation_box_*.png`
- Zipf distributions: `visualizations/token_zipf_*.png`
- Rare tokens: `visualizations/token_rare_topk_*.png`
