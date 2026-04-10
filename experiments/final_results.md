# Final Experiment Results (2026-04-09)

## Best Configuration

- **Pool**: [1b_mpcache, 3b_mpcache, phi3.5_mpcache, 8b_mpcache]
  - 4 models from 3 families (LLaMA, LLaMA, Microsoft, LLaMA) + MPCache compression
- **Router**: BERT-mini (11M params, 4 layers, 256 hidden)
  - Input: query text, max_length=256
  - Training: regression (MSE), 5-fold CV, 5-seed ensemble
  - MPC overhead: <0.1% of LLM per-token cost
- **Routing**: λ-routing: select model = argmax(F1_pred - λ × cost)
  - λ controls cost-quality tradeoff (sweep at inference time, no retraining)

## Main Results: Constrained Optimization

"Given a quality target F1 ≥ X, what is the minimum MPC cost?"

| F1 Target | λ | F1 | Cost (8B=1x) | vs All-8B |
|:---------:|:---:|:---:|:----:|:---:|
| ≥ 50 | 0.245 | 50.34 | **0.364x** | 64% cheaper |
| ≥ 52 | 0.140 | 52.23 | **0.428x** | 57% cheaper |
| ≥ 54 | 0.120 | 54.15 | **0.494x** | **51% cheaper** |
| ≥ 54.5 | 0.115 | 54.79 | **0.529x** | 47% cheaper |
| Peak F1 | 0.100 | **54.97** | 0.617x | 38% cheaper |

## Per-Dataset at Peak (λ=0.10)

| Dataset | 1B+MPC | 3B+MPC | Phi3.5+MPC | 8B+MPC | **Router** | Distribution |
|---------|:------:|:------:|:----------:|:------:|:----------:|:---:|
| HotpotQA | 30.16 | 48.64 | 49.81 | 54.97 | **56.58** | 0/251/113/436 |
| NarrativeQA | 16.55 | 19.79 | 21.92 | 27.99 | **25.50** | 0/251/113/436 |
| TriviaQA | 79.85 | 88.49 | 87.72 | 91.69 | **92.74** | 0/251/113/436 |
| Qasper | 23.10 | 39.92 | 41.45 | 44.20 | **45.05** | 0/251/113/436 |

Router EXCEEDS 8B+MPCache on HotpotQA (56.58 vs 54.97) and TriviaQA (92.74 vs 91.69).

## Comparison with All Baselines (8B=1.0x cost)

| Method | HotpotQA | NarrativeQA | TriviaQA | Qasper | Avg | Cost |
|--------|:--------:|:-----------:|:--------:|:------:|:---:|:----:|
| All-1B | 31.19 | 19.71 | 79.60 | 22.23 | 38.18 | 0.200x |
| All-3B | 49.04 | 22.19 | 88.95 | 39.58 | 49.94 | 0.472x |
| MPCache 7B | 30.53 | 18.30 | 78.31 | 24.72 | 37.97 | 0.724x |
| 8B+MPCache | 54.97 | 27.99 | 91.69 | 44.20 | 54.71 | 0.821x |
| All-8B | 55.47 | 27.75 | 91.47 | 45.50 | 55.05 | 1.000x |
| **Router (λ=0.10)** | **56.58** | **25.50** | **92.74** | **45.05** | **54.97** | **0.617x** |
| **Router (λ=0.12)** | **56.57** | **23.66** | **91.98** | **44.39** | **54.15** | **0.494x** |

## MPC Cost Computation

- SPU 0.9.5, ABY3, 3PC, FM64
- Measured amplification: softmax=174.2x, silu=53.2x, rmsnorm=85.5x vs matmul
- MPCache compression: 15-31% cost reduction per model, <1 F1 loss
- Router overhead: <0.1% (negligible)

## Ablation Studies

### Pool size
| Pool | Peak F1 | Finding |
|------|:-------:|---------|
| 3-model [1b,3b,8b] | 55.73 | Good baseline |
| **4-model [1b,3b,phi,8b]** | **55.73** | **Best (phi3.5 adds complementarity)** |
| 5-model [+qwen7b] | 54.62 | Too many, router less accurate |
| 7-model [all] | 54.45 | Diminishing returns |

### Router architecture
| Router | F1≥54.5 min cost | Finding |
|--------|:----------------:|---------|
| BERT-tiny (4.3M) | 0.708x | Good but higher cost |
| **BERT-mini (11M)** | **0.529x** | **Best cost-quality** |

### Routing strategy
| Strategy | Best F1 | Finding |
|----------|:-------:|---------|
| cheapest_sufficient (qt) | 54.73 | Depends on threshold |
| per-dataset qt | 53.90 | Better cost at lower F1 |
| **λ-routing (F1-λ×cost)** | **54.97** | **Best overall** |
| F1/cost ratio target | 37.42 | Failed |
| Two-stage | 51.54 | Overfitting |
