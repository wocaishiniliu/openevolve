# Router Architecture Comparison

## Classification Router (3-class, 5-fold CV, pool=[1b,3b,8b])

| Router | Params | Accuracy | Easy(1B) F1 | Medium(3B) F1 | Hard(7B) F1 |
|--------|--------|:--------:|:-----------:|:-------------:|:-----------:|
| BERT-tiny | 4.3M | 57.9% | 0.616 | 0.283 | 0.665 |
| BERT-mini | 11M | 59.8% | 0.650 | 0.220 | 0.678 |
| BERT-small | 28M | 59.4% | 0.664 | 0.296 | 0.654 |
| MiniLM-L6 | 22M | 60.4% | 0.656 | 0.136 | 0.680 |

**Finding**: Larger routers provide marginal accuracy gain (+2.5%). Medium(3B) class always poorly classified. Bottleneck is not architecture.

## Regression Router (predict F1, 5-fold CV, pool=[1b,3b,8b], qt=0.90)

| Router | MAE (avg) | Routed F1 | MPC Cost | Score |
|--------|:---------:|:---------:|:--------:|:-----:|
| BERT-tiny | 0.293 | 52.31 | 0.689x | 45.42 |
| BERT-mini | 0.291 | 50.77 | 0.583x | 44.94 |
| BERT-small | 0.299 | 51.68 | 0.660x | 44.93 |
| MiniLM-L6 | 0.281 | 53.23 | 0.857x | 44.66 |

**Finding**: BERT-tiny is surprisingly competitive. MAE ~0.29 across all architectures.

## MPC Overhead (SPU simulator, seq_len=128)

| Router | SPU Time | vs LLM 7B per token |
|--------|:--------:|:-------------------:|
| BERT-tiny | 2.64s | 0.02% |
| BERT-mini | 8.65s | 0.07% |
| BERT-small | 15.99s | 0.14% |

**Finding**: All routers < 0.15% of LLM cost. Not a constraint.
