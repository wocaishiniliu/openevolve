# Label Strategy Comparison

All experiments: BERT-tiny, pool=[1b,3b,8b], max_length=128, 20 epochs, qt=0.90.

## Strategies Tested

### 1. Regression + MSE (baseline)
Predict raw F1 scores per model. Pick cheapest model with predicted F1 >= qt * best.

| qt | F1 | Cost | Score |
|:---:|:---:|:----:|:-----:|
| 0.70 | 47.48 | 0.430x | 43.18 |
| 0.80 | 50.15 | 0.554x | 44.61 |
| 0.90 | 52.31 | 0.689x | 45.42 |
| 1.00 | 53.82 | 0.819x | 45.63 |

### 2. Regression + Weighted MSE
MSE weighted by inverse model cost (cheap model errors penalized more).

| qt=0.90 | F1=50.77 | Cost=0.583x | Score=44.94 |

### 3. Soft Binary + BCE
Predict P(model sufficient) using sigmoid labels: target = sigmoid(temp * (score - threshold)).

| Threshold | Temperature | F1 (qt=0.90) | Cost | Score |
|:---------:|:-----------:|:------------:|:----:|:-----:|
| 0.3 | 3.0 | 49.56 | 0.635x | 43.21 |
| 0.3 | 7.0 | 50.04 | 0.716x | 42.89 |
| 0.5 | 3.0 | 49.99 | 0.702x | 42.96 |
| 0.5 | 7.0 | 50.48 | 0.784x | 42.64 |

### 4. Ranking + Margin Ranking Loss
Predict pairwise model preferences. Margin=0.1.

| qt=0.90 | F1=50.39 | Cost=0.793x | Score=42.46 |

## Summary

| Strategy | Best Score (qt=0.90) |
|----------|:-------------------:|
| **Regression + MSE** | **45.42** |
| Regression + Weighted MSE | 44.94 |
| Soft Binary + BCE (best) | 43.21 |
| Ranking + Margin | 42.46 |

**Finding**: Regression + MSE is the best strategy. Soft binary and ranking did not outperform. This differs from Hybrid LLM (ICLR'24) results, possibly because our task is simpler (4 datasets, clear F1 metric) vs their open-ended generation evaluation.
