# Model Pool Baselines

All models evaluated on 4 LongBench datasets (200 samples each).
F1 scores with post-processing (triviaqa: take first line).

## Per-Model Scores

| Model | Series | MPC Cost | hotpotqa | narrativeqa | triviaqa | qasper | Average |
|-------|--------|:--------:|:--------:|:-----------:|:--------:|:------:|:-------:|
| Llama-3.2-1B | LLaMA | 0.222x | 31.19 | 19.71 | 79.60 | 22.23 | 38.18 |
| Qwen2.5-1.5B | Qwen | 0.255x | 42.60 | 19.18 | 82.11 | 32.55 | 44.11 |
| Llama-3.2-3B | LLaMA | 0.523x | 49.04 | 22.19 | 88.95 | 39.58 | 49.94 |
| MPCache longchat-7B | LLaMA | 0.802x | 30.53 | 18.30 | 78.31 | 24.72 | 37.97 |
| longchat-7B (no compress) | LLaMA | 1.000x | 33.33 | 20.69 | 83.17 | 28.64 | 41.46 |
| Llama-3.1-8B | LLaMA | 1.107x | 55.47 | 27.75 | 91.47 | 45.50 | 55.05 |
| Qwen2.5-14B | Qwen | 2.066x | 62.39 | 28.58 | 90.69 | 45.48 | 56.79 |

## Excluded Models

| Model | Reason |
|-------|--------|
| Gemma-2-2B-it | max_position=8192, too short for LongBench |
| Llama-2-13B-chat | max_position=4096, outputs garbage |

## Selected Model Pool

**[1b, qwen1.5b, 3b, 8b]**: cross-series (LLaMA + Qwen), good cost gradient (0.22x → 1.11x), all support 32K+ context.

## MPC Costs (SPU-measured amplification)

Measured on SPU 0.9.5 simulator (ABY3, 3PC, FM64):
- matmul: 1.0x
- softmax: 174.2x
- silu: 53.2x
- rmsnorm: 85.5x

longchat-7B no-compression = 1.000x baseline.
