"""
Eval worker v3: router sees query + context head/tail.
Input to BERT: "[CLS] query [SEP] context_head ... context_tail [SEP]"
"""
import argparse, json, os, sys, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import StratifiedKFold
from datasets import load_dataset

MPCACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "MPCache")
sys.path.insert(0, MPCACHE_DIR)
from metrics import qa_f1_score

ROUTING_DATASETS = ["hotpotqa", "narrativeqa", "triviaqa", "qasper", "2wikimqa", "musique", "multifieldqa_en"]
POST_PROCESS_DS = ["trec", "triviaqa", "samsum", "lsht"]
DATA_DIR = os.path.join(MPCACHE_DIR, "router", "data")
MPC_COSTS = {"1b": 0.222, "qwen1.5b": 0.255, "3b": 0.523, "8b": 1.107}


def load_contexts():
    """Load raw contexts from LongBench."""
    contexts = {}
    for ds in ROUTING_DATASETS:
        data = load_dataset('THUDM/LongBench', ds, split='test', trust_remote_code=True)
        contexts[ds] = [d['context'] for d in data]
    return contexts


def load_predictions(model_keys, contexts):
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
        ds_contexts = contexts.get(ds, [])
        for i in range(n):
            scores = []
            for mk in model_keys:
                pred = preds_by_model[mk][i]["pred"]
                if ds in POST_PROCESS_DS:
                    pred = pred.lstrip('\n').split('\n')[0]
                scores.append(max(qa_f1_score(pred, gt) for gt in preds_by_model[mk][i]["answers"]))
            ctx = ds_contexts[i] if i < len(ds_contexts) else ""
            samples.append({
                "query": preds_by_model[model_keys[0]][i].get("input", ""),
                "context": ctx,
                "scores": scores,
                "dataset": ds,
            })
    return samples


class RouterDatasetWithContext(Dataset):
    """Tokenizes query + context_head + context_tail into max_length tokens."""
    def __init__(self, samples, tokenizer, max_length=512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        query = item["query"]
        context = item["context"]

        # Strategy: give query full space, split remaining between context head and tail
        query_tokens = self.tokenizer.encode(query, add_special_tokens=False)
        max_query = min(len(query_tokens), self.max_length // 4)  # at most 25% for query
        query_tokens = query_tokens[:max_query]

        remaining = self.max_length - max_query - 3  # [CLS], [SEP], [SEP]
        ctx_tokens = self.tokenizer.encode(context, add_special_tokens=False)

        if len(ctx_tokens) <= remaining:
            ctx_used = ctx_tokens
        else:
            head_len = remaining // 2
            tail_len = remaining - head_len
            ctx_used = ctx_tokens[:head_len] + ctx_tokens[-tail_len:]

        # Build: [CLS] query [SEP] context [SEP]
        cls_id = self.tokenizer.cls_token_id or 101
        sep_id = self.tokenizer.sep_token_id or 102
        input_ids = [cls_id] + query_tokens + [sep_id] + ctx_used + [sep_id]
        attention_mask = [1] * len(input_ids)

        # Pad
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids += [0] * pad_len
            attention_mask += [0] * pad_len
        else:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "targets": torch.tensor(item["targets"], dtype=torch.float32),
        }


class SimpleRouter(torch.nn.Module):
    def __init__(self, model_name, num_outputs):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.head = torch.nn.Linear(self.bert.config.hidden_size, num_outputs)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.head(out.last_hidden_state[:, 0, :])


def make_labels(samples, config):
    for s in samples:
        s["targets"] = s["scores"]
        best_cheap = max(s["scores"][:-1]) if len(s["scores"]) > 1 else s["scores"][0]
        s["label"] = 0 if best_cheap >= 0.5 else (1 if best_cheap >= 0.2 else 2)
    return samples


def get_loss_fn(config, costs):
    loss_type = config.get("loss_type", "mse")
    if loss_type == "weighted_mse":
        inv_costs = torch.tensor([1.0 / c for c in costs])
        inv_costs = inv_costs / inv_costs.sum()
        return lambda p, t: ((p - t) ** 2 * inv_costs.to(p.device)).mean()
    return torch.nn.MSELoss()


def route_samples(preds, config, costs):
    strategy = config.get("routing_strategy", "cheapest_sufficient")
    threshold = config.get("quality_threshold", 0.9)
    choices = []
    for p in preds:
        best = p.max()
        thresh = threshold * best
        chosen = None
        for j in np.argsort(costs):
            if p[j] >= thresh:
                chosen = j
                break
        if chosen is None:
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

    print("Loading contexts from LongBench...", flush=True)
    contexts = load_contexts()
    print("Loading predictions...", flush=True)
    samples = load_predictions(model_keys, contexts)
    if samples is None:
        json.dump({"error": "Failed to load data"}, open(args.output, "w"))
        return
    samples = make_labels(samples, config)

    tokenizer = AutoTokenizer.from_pretrained(config["router_model"])
    loss_fn = get_loss_fn(config, costs)
    labels_arr = np.array([s["label"] for s in samples])
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_preds, all_targets, fold_losses = [], [], []
    ml = config.get("max_length", 512)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(samples, labels_arr)):
        train_data = [samples[i] for i in train_idx]
        val_data = [samples[i] for i in val_idx]

        model = SimpleRouter(config["router_model"], num_models).to(device)
        optimizer = AdamW(model.parameters(), lr=config.get("lr", 2e-5), weight_decay=0.01)
        train_loader = DataLoader(RouterDatasetWithContext(train_data, tokenizer, ml),
                                  batch_size=config.get("batch_size", 16), shuffle=True)
        val_loader = DataLoader(RouterDatasetWithContext(val_data, tokenizer, ml),
                                batch_size=config.get("batch_size", 16))

        best_vl, best_state = float('inf'), None
        for epoch in range(config.get("epochs", 20)):
            model.train()
            for b in train_loader:
                out = model(b["input_ids"].to(device), b["attention_mask"].to(device))
                loss = loss_fn(out, b["targets"].to(device))
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            model.eval()
            vl = 0
            with torch.no_grad():
                for b in val_loader:
                    out = model(b["input_ids"].to(device), b["attention_mask"].to(device))
                    vl += loss_fn(out, b["targets"].to(device)).item()
            avg_vl = vl / len(val_loader)
            if avg_vl < best_vl:
                best_vl = avg_vl
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state); model.to(device).eval()
        fold_losses.append(best_vl)
        with torch.no_grad():
            for b in val_loader:
                out = model(b["input_ids"].to(device), b["attention_mask"].to(device)).cpu().numpy()
                all_preds.append(out)
                all_targets.append(b["targets"].numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    mae = {model_keys[j]: round(float(np.mean(np.abs(all_preds[:, j] - all_targets[:, j]))), 3) for j in range(num_models)}

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
