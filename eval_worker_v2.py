"""
Enhanced eval worker with context_length feature and configurable router.
"""
import argparse, json, os, sys, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import StratifiedKFold

MPCACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "MPCache")
sys.path.insert(0, MPCACHE_DIR)
from metrics import qa_f1_score

ROUTING_DATASETS = ["hotpotqa", "narrativeqa", "triviaqa", "qasper", "2wikimqa", "musique", "multifieldqa_en"]
POST_PROCESS_DS = ["trec", "triviaqa", "samsum", "lsht"]
DATA_DIR = os.path.join(MPCACHE_DIR, "router", "data")
MPC_COSTS = {
    "1b": 0.222, "qwen1.5b": 0.255, "3b": 0.523, "8b": 1.107,
    "1b_mpcache": 0.152, "3b_mpcache": 0.392, "8b_mpcache": 0.909,
}


class RouterWithFeatures(torch.nn.Module):
    """Router that concatenates BERT CLS with extra numeric features."""
    def __init__(self, model_name, num_outputs, num_extra_features=0):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        h = self.bert.config.hidden_size
        self.num_extra = num_extra_features
        if num_extra_features > 0:
            # Project extra features through a small MLP before concatenating
            self.feat_proj = torch.nn.Sequential(
                torch.nn.Linear(num_extra_features, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 32),
            )
            self.head = torch.nn.Linear(h + 32, num_outputs)
        else:
            self.feat_proj = None
            self.head = torch.nn.Linear(h, num_outputs)

    def forward(self, input_ids, attention_mask=None, extra_features=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        if extra_features is not None and self.feat_proj is not None:
            feat = self.feat_proj(extra_features)
            cls = torch.cat([cls, feat], dim=-1)
        return self.head(cls)


class RouterDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length, use_context_length=False):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_context_length = use_context_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        enc = self.tokenizer(
            item["query"], max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        result = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "targets": torch.tensor(item["targets"], dtype=torch.float32),
        }
        if self.use_context_length:
            ctx_feats = item.get("context_features", {})
            if ctx_feats:
                # Use all 14 context features
                feat_values = list(ctx_feats.values())
                result["extra_features"] = torch.tensor(feat_values, dtype=torch.float32)
            else:
                # Fallback: just context_length
                ctx_len = item.get("context_length", 0) / 32000.0
                result["extra_features"] = torch.tensor([ctx_len], dtype=torch.float32)
        return result


def load_predictions(model_keys):
    # Load context features if available
    feat_path = os.path.join(DATA_DIR, "context_features.json")
    ctx_features = {}
    if os.path.exists(feat_path):
        with open(feat_path) as f:
            ctx_features = json.load(f)

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
        ds_feats = ctx_features.get(ds, [])
        for i in range(n):
            scores = []
            for mk in model_keys:
                pred = preds_by_model[mk][i]["pred"]
                if ds in POST_PROCESS_DS:
                    pred = pred.lstrip('\n').split('\n')[0]
                scores.append(max(qa_f1_score(pred, gt) for gt in preds_by_model[mk][i]["answers"]))
            ctx_len = preds_by_model[model_keys[0]][i].get("context_length", 0)
            feats = ds_feats[i] if i < len(ds_feats) else {}
            samples.append({
                "query": preds_by_model[model_keys[0]][i].get("input", ""),
                "scores": scores,
                "dataset": ds,
                "context_length": ctx_len,
                "context_features": feats,
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
    else:
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
        return lambda p, t: ((p - t) ** 2 * inv_costs.to(p.device)).mean()
    elif loss_type == "bce":
        base = torch.nn.BCEWithLogitsLoss()
        return lambda p, t: base(p, t)
    return torch.nn.MSELoss()


def route_samples(preds, config, costs):
    strategy = config.get("routing_strategy", "cheapest_sufficient")
    threshold = config.get("quality_threshold", 0.9)
    choices = []
    for p in preds:
        if strategy == "cheapest_sufficient":
            best = p.max()
            thresh = threshold * best
            chosen = None
            for j in np.argsort(costs):
                if p[j] >= thresh:
                    chosen = j
                    break
            if chosen is None:
                chosen = np.argmax(p)
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
    use_ctx = config.get("use_context_length", False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    samples = load_predictions(model_keys)
    if samples is None:
        json.dump({"error": "Failed to load data"}, open(args.output, "w"))
        return

    # Determine num_extra from actual data
    if use_ctx and samples[0].get("context_features"):
        num_extra = len(samples[0]["context_features"])
    elif use_ctx:
        num_extra = 1
    else:
        num_extra = 0
    samples = make_labels(samples, config)

    tokenizer = AutoTokenizer.from_pretrained(config["router_model"])
    loss_fn = get_loss_fn(config, costs)
    labels_arr = np.array([s["label"] for s in samples])
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_preds, all_targets, fold_losses = [], [], []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(samples, labels_arr)):
        train_data = [samples[i] for i in train_idx]
        val_data = [samples[i] for i in val_idx]

        model = RouterWithFeatures(config["router_model"], num_models, num_extra).to(device)
        optimizer = AdamW(model.parameters(), lr=config.get("lr", 2e-5), weight_decay=0.01)
        ml = config.get("max_length", 128)
        train_loader = DataLoader(RouterDataset(train_data, tokenizer, ml, use_ctx),
                                  batch_size=config.get("batch_size", 16), shuffle=True)
        val_loader = DataLoader(RouterDataset(val_data, tokenizer, ml, use_ctx),
                                batch_size=config.get("batch_size", 16))

        best_vl, best_state = float('inf'), None
        for epoch in range(config.get("epochs", 20)):
            model.train()
            for b in train_loader:
                extra = b.get("extra_features", None)
                if extra is not None:
                    extra = extra.to(device)
                out = model(b["input_ids"].to(device), b["attention_mask"].to(device), extra)
                loss = loss_fn(out, b["targets"].to(device))
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            model.eval()
            vl = 0
            with torch.no_grad():
                for b in val_loader:
                    extra = b.get("extra_features", None)
                    if extra is not None:
                        extra = extra.to(device)
                    out = model(b["input_ids"].to(device), b["attention_mask"].to(device), extra)
                    vl += loss_fn(out, b["targets"].to(device)).item()
            avg_vl = vl / len(val_loader)
            if avg_vl < best_vl:
                best_vl = avg_vl
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state); model.to(device).eval()
        fold_losses.append(best_vl)
        with torch.no_grad():
            for b in val_loader:
                extra = b.get("extra_features", None)
                if extra is not None:
                    extra = extra.to(device)
                out = model(b["input_ids"].to(device), b["attention_mask"].to(device), extra).cpu().numpy()
                all_preds.append(out)
                all_targets.append(b["targets"].numpy())

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
            route_counts[c] += 1; ds_counts[c] += 1
            routed_scores.append(s["scores"][c])
            ds_routed.append(s["scores"][c])
            idx += 1
        per_ds[ds] = {"routed_f1": round(100 * np.mean(ds_routed), 2),
                       "distribution": {model_keys[j]: ds_counts[j] for j in range(num_models)}}

    # Compute MAE
    mae = {model_keys[j]: round(float(np.mean(np.abs(all_preds[:, j] - all_targets[:, j]))), 3) for j in range(num_models)}

    results = {
        "routed_f1": round(100 * np.mean(routed_scores), 2),
        "oracle_f1": round(100 * np.mean([max(s["scores"]) for s in samples]), 2),
        "weighted_cost": round(sum(route_counts[j] / len(samples) * costs[j] for j in range(num_models)), 4),
        "mean_val_loss": round(float(np.mean(fold_losses)), 4),
        "mae": mae,
        "route_distribution": {model_keys[j]: route_counts[j] for j in range(num_models)},
        "per_dataset": per_ds,
    }
    with open(args.output, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
