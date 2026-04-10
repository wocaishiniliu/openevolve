# Best Router Configuration

## Configuration (bert-mini, 256, +ctx_len, 4pool, qt=0.95)

```python
{
    "model_keys": ["1b", "qwen1.5b", "3b", "8b"],
    "router_model": "prajjwal1/bert-mini",
    "max_length": 256,
    "label_type": "regression",
    "loss_type": "mse",
    "lr": 2e-5,
    "epochs": 20,
    "batch_size": 16,
    "routing_strategy": "cheapest_sufficient",
    "quality_threshold": 0.95,
    "use_context_length": True,
}
```

## Pareto Curve Data

| qt | F1 | Cost | Score | 1b | qwen1.5b | 3b | 8b |
|:---:|:---:|:----:|:-----:|:---:|:---:|:---:|:---:|
| 0.70 | 44.32 | 0.373x | 40.59 | 277 | 301 | 146 | 76 |
| 0.80 | 44.51 | 0.459x | 39.92 | 303 | 188 | 154 | 155 |
| 0.85 | 48.51 | 0.584x | 42.67 | 152 | 109 | 327 | 212 |
| 0.90 | 49.16 | 0.582x | 43.34 | 138 | 191 | 232 | 239 |
| 0.95 | 51.59 | 0.721x | 44.38 | 53 | 129 | 260 | 358 |
| 1.00 | 52.66 | 0.845x | 44.21 | 40 | 93 | 162 | 505 |

## Comparison to Baselines

| Method | F1 | Cost | F1 Retained |
|--------|:---:|:----:|:-----------:|
| MPCache 7B (prior work) | 37.97 | 0.802x | 69.0% |
| All-8B (no routing) | 55.05 | 1.107x | 100% |
| **Router (qt=0.95)** | **51.59** | **0.721x** | **93.7%** |
| **Router (qt=0.70)** | **44.32** | **0.373x** | **80.5%** |

## Key Improvements Found

1. **max_length 128→256**: +2 F1 (router sees more query context)
2. **context_length feature**: +1-2 F1 (length correlates with difficulty)
3. **4-model pool (add qwen1.5b)**: lower cost at similar F1
4. **bert-mini > bert-tiny**: marginal F1 gain, negligible MPC overhead

## What Didn't Help

- Soft binary + BCE: worse than regression + MSE
- Ranking + margin loss: worst strategy
- BERT-small / MiniLM-L6: no significant gain over BERT-tiny/mini
- cost_penalty_lambda in loss: no clear benefit
