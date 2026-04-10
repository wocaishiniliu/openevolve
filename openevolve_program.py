"""
Router Configuration — evolved by OpenEvolve.

Goal: maximize routed F1 while minimizing MPC cost. Score = F1 - 10 × cost.

Available models (with MPCache compressed variants):
  1b           cost=0.222x   F1≈38.18
  qwen1.5b     cost=0.255x   F1≈44.11
  3b           cost=0.523x   F1≈49.94
  8b           cost=1.107x   F1≈55.05
  1b_mpcache   cost=0.152x   F1≈37.42  (1B with MPCache, -0.76 F1, -31% cost)
  3b_mpcache   cost=0.392x   F1≈49.21  (3B with MPCache, -0.73 F1, -25% cost)
  8b_mpcache   cost=0.909x   F1≈54.71  (8B with MPCache, -0.33 F1, -18% cost)

Strong baseline to beat: 8B+MPCache F1=54.71, cost=0.909x.

Constraints:
- model_keys: subset of available models, must include at least 2
- router_model: one of ["prajjwal1/bert-tiny", "prajjwal1/bert-mini",
                       "prajjwal1/bert-small", "nreimers/MiniLM-L6-H384-uncased"]
- max_length: 128, 256, or 512
- label_type: "regression" (recommended), "soft_binary", "ranking"
- loss_type: "mse", "weighted_mse", "bce" (with soft_binary)
- quality_threshold: float in [0.5, 1.0]
- lr in [1e-6, 1e-3], epochs in [10, 40], batch_size in [8, 16, 32]
"""


# EVOLVE-BLOCK-START
def get_router_config():
    """Return router configuration. OpenEvolve mutates this function."""
    config = {
        # Model pool: try MPCache compressed variants for lower cost
        "model_keys": ["1b_mpcache", "3b_mpcache", "8b_mpcache"],

        # Router architecture
        "router_model": "prajjwal1/bert-tiny",
        "max_length": 256,

        # Label & loss
        "label_type": "regression",
        "soft_label_threshold": 0.5,
        "soft_label_temperature": 5.0,
        "loss_type": "mse",
        "cost_penalty_lambda": 0.0,

        # Training
        "lr": 2e-5,
        "epochs": 20,
        "batch_size": 16,

        # Routing
        "routing_strategy": "cheapest_sufficient",
        "quality_threshold": 1.0,

        # Cost budget (None = unlimited, 0.909 = beat 8B+MPCache, etc.)
        "cost_budget": None,
    }
    return config
# EVOLVE-BLOCK-END
