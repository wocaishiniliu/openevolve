"""
Worker script for router evaluation. Run as separate process to avoid PyTorch fork issues.
Usage: python eval_worker.py --config config.json --output result.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import StratifiedKFold

MPCACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "MPCache")
sys.path.insert(0, MPCACHE_DIR)
from metrics import qa_f1_score

ROUTING_DATASETS = ["hotpotqa", "narrativeqa", "triviaqa", "qasper"]
POST_PROCESS_DS = ["trec", "triviaqa", "samsum", "lsht"]
DATA_DIR = os.path.join(MPCACHE_DIR, "router", "data")
MPC_COSTS = {"1b": 0.222, "qwen1.5b": 0.255, "3b": 0.523, "8b": 1.107}


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


class SimpleRouter(torch.nn.Module):
    def __init__(self, model_name, num_outputs):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.head = torch.nn.Linear(self.bert.config.hidden_size, num_outputs)

    def forward(self, input_ids, attention_mask=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.head(out.last_hidden_state[:, 0, :])


def load_predictions(model_keys):
    samples = []
    for ds in ROUTING_DATASETS:
        preds_by_model = {}
        for mk in model_keys:
            path = os.path.join(DATA_DIR, mk, f"{ds}.jsonl")
            if not os.path.exists(path):
                return None
            with open(path) as f:
                preds_by_model[mk] = [json.loads(l) for l in f]
        n = min(len(preds_by_model[mk]) for mk in model_keys)
        for i in range(n):
            scores = []
            for mk in model_keys:
                pred = preds_by_model[mk][i]["pred"]
                if ds in POST_PROCESS_DS:
                    pred = pred.lstrip('\n').split('\n')[0]
                scores.append(max(qa_f1_score(pred, gt) for gt in preds_by_model[mk][i]["answers"]))
            samples.append({
                "query": preds_by_model[model_keys[0]][i].get("input", ""),
                "scores": scores, "dataset": ds,
            })
    return samples


def make_labels(samples, config):
    label_type = config.get("label_type", "regression")
    if label_type == "regression":
        for s in samples:
            s["targets"] = s["scores"]
    elif label_type == "soft_binary":
        threshold = config.get("soft_label_threshold", 0.5)
        temperature = config.get("soft_label_temperature", 5.0)
        for s in samples:
            s["targets"] = [1.0 / (1.0 + np.exp(-temperature * (sc - threshold))) for sc in s["scores"]]
    elif label_type == "ranking":
        for s in samples:
            s["targets"] = s["scores"]
    for s in samples:
        best_cheap = max(s["scores"][:-1]) if len(s["scores"]) > 1 else s["scores"][0]
        s["label"] = 0 if best_cheap >= 0.5 else (1 if best_cheap >= 0.2 else 2)
    return samples


def get_loss_fn(config, costs):
    loss_type = config.get("loss_type", "mse")
    if loss_type == "weighted_mse":
        inv_costs = torch.tensor([1.0 / c for c in costs])
        inv_costs = inv_costs / inv_costs.sum()
        def loss_fn(preds, targets):
            return ((preds - targets) ** 2 * inv_costs.to(preds.device)).mean()
        return loss_fn
    elif loss_type == "bce":
        base = torch.nn.BCEWithLogitsLoss()
        return lambda p, t: base(p, t)
    elif loss_type == "margin_ranking":
        def loss_fn(preds, targets):
            loss, count = 0.0, 0
            for i in range(preds.shape[1]):
                for j in range(i + 1, preds.shape[1]):
                    sign = torch.sign(targets[:, i] - targets[:, j])
                    loss += torch.nn.functional.margin_ranking_loss(preds[:, i], preds[:, j], sign, margin=0.1)
                    count += 1
            return loss / max(count, 1)
        return loss_fn
    else:
        return torch.nn.MSELoss()


def route_samples(preds, config, costs):
    strategy = config.get("routing_strategy", "cheapest_sufficient")
    threshold = config.get("quality_threshold", 0.9)
    choices = []
    for i in range(len(preds)):
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
            budget = costs[0] + threshold * (costs[-1] - costs[0])
            chosen, best_score = None, -1
            for j in range(len(costs)):
                if costs[j] <= budget and p[j] > best_score:
                    best_score = p[j]
                    chosen = j
            if chosen is None:
                chosen = 0
        else:
            chosen = np.argmax(p)
        choices.append(chosen)
    return choices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    model_keys = config["model_keys"]
    costs = [MPC_COSTS[mk] for mk in model_keys]
    num_models = len(model_keys)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    samples = load_predictions(model_keys)
    if samples is None:
        json.dump({"error": "Failed to load data"}, open(args.output, "w"))
        return
    samples = make_labels(samples, config)

    tokenizer = AutoTokenizer.from_pretrained(config["router_model"])
    loss_fn = get_loss_fn(config, costs)
    labels_arr = np.array([s["label"] for s in samples])
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_preds, all_targets, fold_losses = [], [], []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(samples, labels_arr)):
        train_data = [samples[i] for i in train_idx]
        val_data = [samples[i] for i in val_idx]

        model = SimpleRouter(config["router_model"], num_models).to(device)
        optimizer = AdamW(model.parameters(), lr=config.get("lr", 2e-5), weight_decay=0.01)
        train_ds = RouterDataset(train_data, tokenizer, config.get("max_length", 128))
        val_ds = RouterDataset(val_data, tokenizer, config.get("max_length", 128))
        train_loader = DataLoader(train_ds, batch_size=config.get("batch_size", 16), shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config.get("batch_size", 16))

        best_val_loss, best_state = float('inf'), None
        for epoch in range(config.get("epochs", 10)):
            model.train()
            for batch in train_loader:
                out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                loss = loss_fn(out, batch["targets"].to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            model.eval()
            vl = 0
            with torch.no_grad():
                for batch in val_loader:
                    out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                    vl += loss_fn(out, batch["targets"].to(device)).item()
            avg_vl = vl / len(val_loader)
            if avg_vl < best_val_loss:
                best_val_loss = avg_vl
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        model.to(device).eval()
        fold_losses.append(best_val_loss)
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device)).cpu().numpy()
                all_preds.append(out)
                all_targets.append(batch["targets"].numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    choices = route_samples(all_preds, config, costs)
    routed_scores, route_counts = [], [0] * num_models
    per_ds = {}
    idx = 0
    for ds in ROUTING_DATASETS:
        ds_samples = [s for s in samples if s["dataset"] == ds]
        ds_routed, ds_counts = [], [0] * num_models
        for s in ds_samples:
            c = choices[idx]
            route_counts[c] += 1
            ds_counts[c] += 1
            routed_scores.append(s["scores"][c])
            ds_routed.append(s["scores"][c])
            idx += 1
        per_ds[ds] = {"routed_f1": round(100 * np.mean(ds_routed), 2),
                       "distribution": {model_keys[j]: ds_counts[j] for j in range(num_models)}}

    results = {
        "routed_f1": round(100 * np.mean(routed_scores), 2),
        "oracle_f1": round(100 * np.mean([max(s["scores"]) for s in samples]), 2),
        "weighted_cost": round(sum(route_counts[j] / len(samples) * costs[j] for j in range(num_models)), 4),
        "mean_val_loss": round(float(np.mean(fold_losses)), 4),
        "mae": {model_keys[j]: round(float(np.mean(np.abs(all_preds[:, j] - all_targets[:, j]))), 3) for j in range(num_models)},
        "route_distribution": {model_keys[j]: route_counts[j] for j in range(num_models)},
        "per_dataset": per_ds,
    }
    with open(args.output, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
