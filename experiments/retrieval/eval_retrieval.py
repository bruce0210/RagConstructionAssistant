# eval_retrieval.py
import json
from pathlib import Path
from collections import defaultdict
from typing import List

import numpy as np
from tqdm import tqdm

from search import search

ROOT = Path(__file__).resolve().parent
DEV_QUERIES = ROOT / "data" / "dev_queries.jsonl"
DEV_LABELS = ROOT / "data" / "dev_labels.json"

def load_queries():
    qs = []
    with DEV_QUERIES.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            qs.append(json.loads(line))
    return qs

def load_labels():
    with DEV_LABELS.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    # key: int(qid), val: List[int]
    return {int(k): v for k, v in raw.items()}

def hit_at_k(ranked_ids: List[int], gold_ids: List[int], k: int) -> float:
    return 1.0 if any(i in gold_ids for i in ranked_ids[:k]) else 0.0

def ndcg_at_k(ranked_ids: List[int], gold_ids: List[int], k: int) -> float:
    dcg = 0.0
    for rank, doc_id in enumerate(ranked_ids[:k], start=1):
        rel = 1.0 if doc_id in gold_ids else 0.0
        if rel > 0:
            dcg += rel / np.log2(rank + 1)
    ideal_rels = [1.0] * min(len(gold_ids), k)
    idcg = sum(rel / np.log2(rank + 1) for rank, rel in enumerate(ideal_rels, start=1))
    return dcg / idcg if idcg > 0 else 0.0

def main():
    queries = load_queries()
    labels = load_labels()

    metrics = defaultdict(list)

    for q in tqdm(queries, desc="Evaluating"):
        qid = q["id"]
        if qid not in labels:
            # 没有 gold（可能还没标完），跳过
            continue
        gold_ids = labels[qid]

        res = search(q["query"], top_k=10)
        ranked_ids = [r["id"] for r in res]

        metrics["hit@1"].append(hit_at_k(ranked_ids, gold_ids, 1))
        metrics["hit@3"].append(hit_at_k(ranked_ids, gold_ids, 3))
        metrics["hit@10"].append(hit_at_k(ranked_ids, gold_ids, 10))
        metrics["ndcg@10"].append(ndcg_at_k(ranked_ids, gold_ids, 10))

    print("\n=== Retrieval Metrics on dev set (bge-m3 + FAISS) ===")
    for name, vals in metrics.items():
        print(f"{name}: {np.mean(vals):.4f}  (n={len(vals)})")

if __name__ == "__main__":
    main()
