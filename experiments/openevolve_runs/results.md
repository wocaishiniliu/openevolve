# OpenEvolve Optimization Runs

## Run 1: 30 iterations (CUDA fork issues)
- All 30 mutations failed due to PyTorch CUDA fork error
- Only initial program evaluated successfully
- Fix: use subprocess worker (eval_worker.py)

## Run 2: 30 iterations (CPU, 10 epochs)
- 47 successful evaluations, 0 failures
- Initial: score=43.25 (F1=50.44, cost=0.719x)
- Best: score=43.38 (F1=50.32, cost=0.695x)
- Improvement: +0.13 score (marginal)
- Found: weighted_mse slightly better than mse
- Problem: weak baseline due to epochs=10

## Run 3: 60 iterations (CPU, 20 epochs, enhanced system message)
- 47 successful evaluations, 0 failures  
- Initial: score=42.88 (F1=50.17, cost=0.729x)
- **Best: score=45.19 (F1=52.77, cost=0.758x)**
- Improvement: **+2.31 score, +2.60 F1**
- Found best config:
  - model_keys: [1b, 3b, 8b]
  - router_model: bert-tiny
  - **max_length: 256** (key finding)
  - label_type: regression
  - loss_type: mse
  - quality_threshold: 0.9
- LLM explored: 20x regression, 1x soft_binary, 9x weighted_mse, 1x bce
- LLM did NOT explore ranking despite system message guidance

## Key OpenEvolve Findings
1. max_length 128→256 is the most impactful single change
2. LLM mutations are conservative - prefer small parameter tweaks over strategy changes
3. Subprocess worker essential to avoid PyTorch fork issues
4. Each evaluation ~2 min on CPU, allowing 60 iterations in ~2 hours
