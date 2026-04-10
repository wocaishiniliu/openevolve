"""
Extract context statistical features from LongBench datasets.
Save to router/data/context_features.json for router training.
"""
import json
import re
import os
import numpy as np
from datasets import load_dataset

ROUTING_DATASETS = ["hotpotqa", "narrativeqa", "triviaqa", "qasper", "2wikimqa", "musique", "multifieldqa_en"]
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "MPCache", "router", "data", "context_features.json")


def extract_features(context, query):
    """Extract statistical features from context and query."""
    # Context features
    ctx_len_chars = len(context)
    ctx_len_words = len(context.split())
    num_sentences = len(re.split(r'[.!?]+', context))
    num_paragraphs = len([p for p in context.split('\n\n') if p.strip()])
    num_newlines = context.count('\n')

    # Vocabulary diversity
    words = context.lower().split()
    vocab_size = len(set(words)) if words else 0
    vocab_diversity = vocab_size / len(words) if words else 0

    # Avg word length
    avg_word_len = np.mean([len(w) for w in words]) if words else 0

    # Contains special content
    has_table = 1.0 if ('|' in context and context.count('|') > 5) else 0.0
    has_code = 1.0 if ('def ' in context or 'class ' in context or '```' in context) else 0.0
    has_numbers = sum(1 for c in context if c.isdigit()) / max(len(context), 1)

    # Query features
    query_len_words = len(query.split())
    query_has_compare = 1.0 if any(w in query.lower() for w in ['compare', 'difference', 'which', 'between']) else 0.0
    query_has_why = 1.0 if any(w in query.lower() for w in ['why', 'how', 'explain', 'reason']) else 0.0
    query_has_what = 1.0 if any(w in query.lower() for w in ['what', 'who', 'when', 'where']) else 0.0

    # Number of passages (multi-doc indicator)
    num_passages = len(re.findall(r'Passage \d+', context))

    return {
        "ctx_len_chars": ctx_len_chars / 100000,       # normalize
        "ctx_len_words": ctx_len_words / 20000,
        "num_sentences": num_sentences / 500,
        "num_paragraphs": num_paragraphs / 50,
        "vocab_diversity": vocab_diversity,
        "avg_word_len": avg_word_len / 10,
        "has_table": has_table,
        "has_code": has_code,
        "has_numbers": has_numbers,
        "query_len_words": query_len_words / 50,
        "query_has_compare": query_has_compare,
        "query_has_why": query_has_why,
        "query_has_what": query_has_what,
        "num_passages": num_passages / 20,
    }


def main():
    all_features = {}

    for ds_name in ROUTING_DATASETS:
        print(f"Processing {ds_name}...")
        data = load_dataset('THUDM/LongBench', ds_name, split='test', trust_remote_code=True)

        ds_features = []
        for sample in data:
            feats = extract_features(sample['context'], sample['input'])
            ds_features.append(feats)

        all_features[ds_name] = ds_features
        print(f"  {len(ds_features)} samples, {len(ds_features[0])} features each")

    with open(OUT_PATH, 'w') as f:
        json.dump(all_features, f)
    print(f"\nSaved to {OUT_PATH}")

    # Print feature stats
    print("\nFeature statistics (across all datasets):")
    all_feats_flat = []
    for ds in all_features.values():
        all_feats_flat.extend(ds)
    keys = list(all_feats_flat[0].keys())
    for k in keys:
        vals = [f[k] for f in all_feats_flat]
        print(f"  {k:<20} mean={np.mean(vals):.3f}  std={np.std(vals):.3f}  min={np.min(vals):.3f}  max={np.max(vals):.3f}")


if __name__ == '__main__':
    main()
