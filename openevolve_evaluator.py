"""
OpenEvolve evaluator for router optimization.

Trains a router with the evolved configuration and evaluates routing quality.
Uses pre-computed model predictions (no LLM inference needed, ~2-3 min per eval).

Returns:
- score: composite metric (routed_F1 - cost_penalty)
- artifacts: per-dataset scores, routing distribution, MAE, etc.
"""

import os
import sys
import json
import importlib.util
import traceback
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import StratifiedKFold

# Path to MPCache project with pre-computed predictions
MPCACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "MPCache")
sys.path.insert(0, MPCACHE_DIR)
from metrics import qa_f1_score

# Fixed constants
ROUTING_DATASETS = ["hotpotqa", "narrativeqa", "triviaqa", "qasper"]
POST_PROCESS_DS = ["trec", "triviaqa", "samsum", "lsht"]
DATA_DIR = os.path.join(MPCACHE_DIR, "router", "data")

MPC_COSTS = {
    "1b": 0.222,
    "qwen1.5b": 0.255,
    "3b": 0.523,
    "8b": 1.107,
    # MPCache compressed variants
    "1b_mpcache": 0.152,
    "3b_mpcache": 0.392,
    "8b_mpcache": 0.909,
}

# Map model_key to data directory (for predictions)
MODEL_DIR_MAP = {
    "1b": "1b", "qwen1.5b": "qwen1.5b", "3b": "3b", "8b": "8b",
    "1b_mpcache": "1b_mpcache",
    "3b_mpcache": "3b_mpcache",
    "8b_mpcache": "8b_mpcache",
}

# Baselines for reference (all-8B scores)
ALL_8B_F1 = 55.05
ALL_8B_COST = 1.107
BB_MPCACHE_F1 = 54.71
BB_MPCACHE_COST = 0.909


# === Data Loading ===

def load_predictions(model_keys):
    """Load pre-computed predictions and compute per-sample F1 scores."""
    samples = []
    for dataset_name in ROUTING_DATASETS:
        preds_by_model = {}
        for mk in model_keys:
            path = os.path.join(DATA_DIR, mk, f"{dataset_name}.jsonl")
            if not os.path.exists(path):
                print(f"[eval] WARNING: {path} not found")
                return None
            with open(path) as f:
                preds_by_model[mk] = [json.loads(line) for line in f]

        n = min(len(preds_by_model[mk]) for mk in model_keys)
        for i in range(n):
            scores = []
            for mk in model_keys:
                pred = preds_by_model[mk][i]["pred"]
                if dataset_name in POST_PROCESS_DS:
                    pred = pred.lstrip('\n').split('\n')[0]
                answers = preds_by_model[mk][i]["answers"]
                score = max(qa_f1_score(pred, gt) for gt in answers)
                scores.append(score)

            query = preds_by_model[model_keys[0]][i].get("input", "")
            samples.append({
                "query": query,
                "scores": scores,
                "dataset": dataset_name,
            })
    return samples


# === Label Strategies ===

def make_labels(samples, config):
    """Create training targets based on label_type."""
    label_type = config.get("label_type", "regression")
    model_keys = config["model_keys"]

    if label_type == "regression":
        # Raw F1 scores as targets
        for s in samples:
            s["targets"] = s["scores"]

    elif label_type == "soft_binary":
        threshold = config.get("soft_label_threshold", 0.5)
        temperature = config.get("soft_label_temperature", 5.0)
        for s in samples:
            # Soft sigmoid: P(model sufficient) = sigmoid(temp * (score - threshold))
            s["targets"] = [
                1.0 / (1.0 + np.exp(-temperature * (sc - threshold)))
                for sc in s["scores"]
            ]

    elif label_type == "ranking":
        # Pairwise: for each pair (i,j), target = 1 if score_i > score_j
        # Store raw scores; loss function handles pairwise comparison
        for s in samples:
            s["targets"] = s["scores"]

    # Assign stratification label (for CV splits)
    for s in samples:
        best_cheap = max(s["scores"][:-1]) if len(s["scores"]) > 1 else s["scores"][0]
        if best_cheap >= 0.5:
            s["label"] = 0
        elif best_cheap >= 0.2:
            s["label"] = 1
        else:
            s["label"] = 2

    return samples


# === Router Model ===

class SimpleRouter(torch.nn.Module):
    def __init__(self, model_name, num_outputs):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.head = torch.nn.Linear(self.bert.config.hidden_size, num_outputs)

    def forward(self, input_ids, attention_mask=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.head(cls)


class RouterDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        enc = self.tokenizer(
            item["query"], max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "targets": torch.tensor(item["targets"], dtype=torch.float32),
        }


# === Loss Functions ===

def get_loss_fn(config, costs):
    """Return loss function based on config."""
    loss_type = config.get("loss_type", "mse")
    lam = config.get("cost_penalty_lambda", 0.0)

    if loss_type == "mse":
        base_loss = torch.nn.MSELoss()
        def loss_fn(preds, targets):
            return base_loss(preds, targets)
        return loss_fn

    elif loss_type == "weighted_mse":
        # Weight errors by inverse cost (cheap model errors matter more)
        inv_costs = torch.tensor([1.0 / c for c in costs], dtype=torch.float32)
        inv_costs = inv_costs / inv_costs.sum()

        def loss_fn(preds, targets):
            diff_sq = (preds - targets) ** 2
            weighted = diff_sq * inv_costs.to(preds.device)
            return weighted.mean()
        return loss_fn

    elif loss_type == "bce":
        base_loss = torch.nn.BCEWithLogitsLoss()
        def loss_fn(preds, targets):
            return base_loss(preds, targets)
        return loss_fn

    elif loss_type == "margin_ranking":
        def loss_fn(preds, targets):
            # Pairwise margin ranking: for all pairs (i,j), if target_i > target_j,
            # then pred_i should be > pred_j
            loss = 0.0
            count = 0
            n_models = preds.shape[1]
            for i in range(n_models):
                for j in range(i + 1, n_models):
                    # target: 1 if score_i > score_j, -1 otherwise
                    sign = torch.sign(targets[:, i] - targets[:, j])
                    margin_loss = torch.nn.functional.margin_ranking_loss(
                        preds[:, i], preds[:, j], sign, margin=0.1
                    )
                    loss += margin_loss
                    count += 1
            return loss / max(count, 1)
        return loss_fn

    else:
        return torch.nn.MSELoss()


# === Routing Logic ===

def route_samples(preds, config, costs):
    """Given predicted scores, decide which model to use for each sample."""
    strategy = config.get("routing_strategy", "cheapest_sufficient")
    threshold = config.get("quality_threshold", 0.9)
    n = len(preds)
    choices = []

    for i in range(n):
        p = preds[i]

        if strategy == "cheapest_sufficient":
            best_pred = p.max()
            thresh = threshold * best_pred
            chosen = None
            for j in np.argsort(costs):
                if p[j] >= thresh:
                    chosen = j
                    break
            if chosen is None:
                chosen = np.argmax(p)

        elif strategy == "cost_constrained":
            # Pick highest predicted score among affordable models
            # Use threshold as cost budget multiplier (relative to cheapest)
            budget = costs[0] + threshold * (costs[-1] - costs[0])
            chosen = None
            best_score = -1
            for j in range(len(costs)):
                if costs[j] <= budget and p[j] > best_score:
                    best_score = p[j]
                    chosen = j
            if chosen is None:
                chosen = 0  # cheapest fallback

        else:
            chosen = np.argmax(p)

        choices.append(chosen)
    return choices


# === Training ===

def train_and_evaluate(samples, config, device):
    """Train router with 5-fold CV and evaluate routing."""
    model_keys = config["model_keys"]
    costs = [MPC_COSTS[mk] for mk in model_keys]
    num_models = len(model_keys)

    tokenizer = AutoTokenizer.from_pretrained(config["router_model"])
    loss_fn = get_loss_fn(config, costs)

    labels_arr = np.array([s["label"] for s in samples])
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_preds = []
    all_targets = []
    fold_losses = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(samples, labels_arr)):
        train_data = [samples[i] for i in train_idx]
        val_data = [samples[i] for i in val_idx]

        model = SimpleRouter(config["router_model"], num_models).to(device)
        optimizer = AdamW(model.parameters(), lr=config.get("lr", 2e-5),
                          weight_decay=0.01)

        train_ds = RouterDataset(train_data, tokenizer, config.get("max_length", 128))
        val_ds = RouterDataset(val_data, tokenizer, config.get("max_length", 128))
        train_loader = DataLoader(train_ds, batch_size=config.get("batch_size", 16), shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config.get("batch_size", 16))

        best_val_loss = float('inf')
        best_state = None

        for epoch in range(config.get("epochs", 20)):
            model.train()
            for batch in train_loader:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                tgt = batch["targets"].to(device)
                out = model(ids, mask)
                loss = loss_fn(out, tgt)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    ids = batch["input_ids"].to(device)
                    mask = batch["attention_mask"].to(device)
                    tgt = batch["targets"].to(device)
                    out = model(ids, mask)
                    val_loss += loss_fn(out, tgt).item()
            avg_val = val_loss / len(val_loader)
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        model.to(device).eval()
        fold_losses.append(best_val_loss)

        with torch.no_grad():
            for batch in val_loader:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                tgt = batch["targets"]
                out = model(ids, mask).cpu().numpy()
                all_preds.append(out)
                all_targets.append(tgt.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # For soft_binary, convert back to score-space for routing evaluation
    if config.get("label_type") == "soft_binary":
        # Targets were sigmoid-transformed; use raw scores for evaluation
        raw_scores = np.array([s["scores"] for s in samples])
        # Reorder to match CV order
        all_raw_scores = raw_scores  # approximate: samples order matches
    else:
        all_raw_scores = all_targets

    # Evaluate routing
    choices = route_samples(all_preds, config, costs)
    routed_scores = []
    route_counts = [0] * num_models
    per_dataset_results = {}

    idx = 0
    for ds in ROUTING_DATASETS:
        ds_samples = [s for s in samples if s["dataset"] == ds]
        ds_routed = []
        ds_counts = [0] * num_models
        for s in ds_samples:
            c = choices[idx]
            route_counts[c] += 1
            ds_counts[c] += 1
            routed_scores.append(s["scores"][c])
            ds_routed.append(s["scores"][c])
            idx += 1
        per_dataset_results[ds] = {
            "routed_f1": round(100 * np.mean(ds_routed), 2),
            "distribution": {model_keys[j]: ds_counts[j] for j in range(num_models)},
        }

    avg_routed_f1 = round(100 * np.mean(routed_scores), 2)
    oracle_f1 = round(100 * np.mean([max(s["scores"]) for s in samples]), 2)
    weighted_cost = sum(route_counts[j] / len(samples) * costs[j] for j in range(num_models))
    mae = np.mean(np.abs(all_preds - all_targets), axis=0)

    return {
        "routed_f1": avg_routed_f1,
        "oracle_f1": oracle_f1,
        "weighted_cost": round(weighted_cost, 4),
        "mean_val_loss": round(float(np.mean(fold_losses)), 4),
        "mae": {model_keys[j]: round(float(mae[j]), 3) for j in range(num_models)},
        "route_distribution": {model_keys[j]: route_counts[j] for j in range(num_models)},
        "per_dataset": per_dataset_results,
    }


# === Main Evaluator Entry Point ===

def evaluate(program_path):
    """Called by OpenEvolve. Trains router with evolved config, returns score."""
    try:
        # Load evolved config
        # OpenEvolve may pass a file path or a directory path
        if os.path.isfile(program_path):
            prog_file = program_path
        else:
            prog_file = os.path.join(program_path, "openevolve_program.py")
            if not os.path.exists(prog_file):
                for f in os.listdir(program_path):
                    if f.endswith(".py") and "program" in f:
                        prog_file = os.path.join(program_path, f)
                        break

        spec = importlib.util.spec_from_file_location("evolved", prog_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = module.get_router_config()

        print(f"[eval] Config: {json.dumps(config, indent=2)}")

        # Validate
        model_keys = config.get("model_keys", [])
        if len(model_keys) < 2:
            return {"score": 0.0, "error": "Need at least 2 models in pool"}
        for mk in model_keys:
            if mk not in MPC_COSTS:
                return {"score": 0.0, "error": f"Unknown model: {mk}"}

        valid_routers = ["prajjwal1/bert-tiny", "prajjwal1/bert-mini",
                         "prajjwal1/bert-small", "nreimers/MiniLM-L6-H384-uncased"]
        if config.get("router_model", "") not in valid_routers:
            return {"score": 0.0, "error": f"Invalid router: {config.get('router_model')}"}

        # Run training in a subprocess to avoid PyTorch fork issues
        import subprocess, tempfile
        config_path = tempfile.mktemp(suffix=".json")
        result_path = tempfile.mktemp(suffix=".json")
        with open(config_path, "w") as f:
            json.dump(config, f)

        worker_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_worker.py")
        cmd = [
            sys.executable, worker_script,
            "--config", config_path,
            "--output", result_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=540)
        if proc.returncode != 0:
            return {"score": 0.0, "error": proc.stderr[-500:] if proc.stderr else "worker failed"}

        with open(result_path) as f:
            results = json.load(f)
        os.unlink(config_path)
        os.unlink(result_path)

        # Score depends on cost_budget mode
        cost_budget = config.get("cost_budget", None)
        routed_f1 = results["routed_f1"]
        weighted_cost = results["weighted_cost"]

        if cost_budget is not None:
            # Hard constraint: penalize heavily if over budget
            if weighted_cost > cost_budget:
                score = routed_f1 - 100 * (weighted_cost - cost_budget)
            else:
                score = routed_f1  # within budget, pure F1 maximization
        else:
            # No budget: pure F1 maximization
            score = routed_f1

        print(f"[eval] Routed F1: {routed_f1}, Cost: {weighted_cost}, Budget: {cost_budget}, Score: {score:.2f}")
        print(f"[eval] Per-dataset: {json.dumps(results['per_dataset'], indent=2)}")

        # Build artifacts for LLM feedback
        budget_str = f"{cost_budget}x" if cost_budget else "unlimited"
        over_budget = f" ⚠️ OVER BUDGET by {weighted_cost - cost_budget:.3f}x" if cost_budget and weighted_cost > cost_budget else ""
        artifacts = {
            "stderr": "",
            "results_summary": (
                f"Routed F1: {routed_f1} (oracle: {results['oracle_f1']})\n"
                f"MPC Cost: {weighted_cost:.3f}x (budget: {budget_str}){over_budget}\n"
                f"Route distribution: {results['route_distribution']}\n"
                f"MAE: {results['mae']}\n"
                f"Val loss: {results['mean_val_loss']}\n"
                f"Per-dataset F1: {json.dumps({ds: r['routed_f1'] for ds, r in results['per_dataset'].items()})}\n"
                f"\nBaselines:\n"
                f"  All-3B: F1=49.94, cost=0.523x\n"
                f"  MPCache 7B: F1=37.97, cost=0.802x\n"
                f"  All-8B: F1={ALL_8B_F1}, cost={ALL_8B_COST}x\n"
                f"  This config: F1={routed_f1}, cost={weighted_cost:.3f}x\n"
                f"  F1 retained: {routed_f1/ALL_8B_F1*100:.1f}% at {weighted_cost/ALL_8B_COST*100:.1f}% cost"
            ),
        }

        return {
            "score": round(score, 2),
            "metrics": {
                "performance": routed_f1,
                "cost": weighted_cost,
            },
            "artifacts": artifacts,
        }

    except Exception as e:
        traceback.print_exc()
        return {"score": 0.0, "error": str(e)}
