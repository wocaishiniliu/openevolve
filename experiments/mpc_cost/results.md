# MPC Cost Analysis

## SPU Benchmark (ABY3, 3PC, FM64)

Measured on SPU 0.9.5 simulator with seq_len=32-128.

| Operation | us/element | Relative to matmul |
|-----------|:----------:|:------------------:|
| matmul | 5.77 | 1.0x |
| exp | 363.05 | 62.9x |
| silu | 306.78 | 53.2x |
| rmsnorm | 493.32 | 85.5x |
| softmax | 1004.70 | 174.2x |

Data saved in: `mpc_benchmark/mpc_op_costs.json`

## Model MPC Costs (relative to longchat-7B no-compression = 1.000x)

| Model | MPC Cost |
|-------|:--------:|
| Llama-3.2-1B | 0.222x |
| Qwen2.5-1.5B | 0.255x |
| Llama-3.2-3B | 0.523x |
| longchat-7B + MPCache | 0.802x |
| longchat-7B (no compression) | 1.000x |
| Llama-3.1-8B | 1.107x |
| Qwen2.5-14B | 2.066x |

## MPCache Compression Effectiveness

7B cost breakdown: linear 91.0%, softmax 8.5%, silu 0.3%, rmsnorm 0.1%.
MPCache compresses cache to ~5% but only saves ~20% total MPC cost (FFN dominates).
Router saves 30-65% by avoiding expensive models entirely.

## Router MPC Overhead

| Router | SPU Time | vs 7B per token | vs 7B × 100 tokens |
|--------|:--------:|:---------------:|:-------------------:|
| BERT-tiny | 2.64s | 0.02% | 0.000% |
| BERT-mini | 8.65s | 0.07% | 0.001% |
| BERT-small | 15.99s | 0.14% | 0.001% |

Router overhead is negligible for all architectures tested.
