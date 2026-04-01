"""
Router Configuration — evolved by OpenEvolve.

OpenEvolve will mutate get_router_config() to explore different router strategies.
The evaluator trains a router using these settings and measures routed F1 vs MPC cost.

Fixed resources (not evolved):
- Pre-computed predictions for all models: MPCache/router/data/{1b,qwen1.5b,3b,8b}/*.jsonl
- MPC costs: 1b=0.222, qwen1.5b=0.255, 3b=0.523, 8b=1.107 (relative to longchat-7B)

Constraints:
- model_keys: subset of ["1b", "qwen1.5b", "3b", "8b"], must include at least 2
- router_model: one of ["prajjwal1/bert-tiny", "prajjwal1/bert-mini", "prajjwal1/bert-small", "nreimers/MiniLM-L6-H384-uncased"]
- max_length: one of [128, 256, 512]
- label_type: one of ["regression", "soft_binary", "ranking"]
- loss_type: must be compatible with label_type (see below)
- quality_threshold: float in [0.5, 1.0], controls cost-quality tradeoff at routing time
- lr: float in [1e-6, 1e-3]
- epochs: int in [5, 40]
- batch_size: one of [8, 16, 32]
- cost_penalty_lambda: float in [0.0, 1.0], penalty for routing to expensive models

Label-Loss compatibility:
- regression + mse: predict raw F1 scores per model
- regression + weighted_mse: weight MSE by inverse model cost (penalize cheap-model errors more)
- soft_binary + bce: predict P(model is sufficient), soft sigmoid labels
- ranking + margin_ranking: predict pairwise preferences between models

Routing strategy at inference time:
- "cheapest_sufficient": among models with predicted score >= threshold * best, pick cheapest
- "cost_constrained": among models with cost <= budget, pick highest predicted score
"""


# EVOLVE-BLOCK-START
def get_router_config():
    """Return the full router configuration. OpenEvolve will evolve this function."""
    config = {
        # Model pool: which models to route between
        "model_keys": ["1b", "3b", "8b"],

        # Router architecture
        "router_model": "prajjwal1/bert-tiny",
        "max_length": 128,

        # Label strategy
        "label_type": "regression",

        # Soft binary params (only used if label_type == "soft_binary")
        "soft_label_threshold": 0.5,   # F1 threshold for "sufficient"
        "soft_label_temperature": 5.0, # sigmoid sharpness

        # Loss function
        "loss_type": "mse",
        "cost_penalty_lambda": 0.0,

        # Training hyperparameters
        "lr": 2e-5,
        "epochs": 20,
        "batch_size": 16,

        # Routing decision at inference time
        "routing_strategy": "cheapest_sufficient",
        "quality_threshold": 0.9,
    }
    return config
# EVOLVE-BLOCK-END
