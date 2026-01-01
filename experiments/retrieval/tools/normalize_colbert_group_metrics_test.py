import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

DETAILS = DATA / "test_colbert_details.jsonl"
OUT     = DATA / "test_colbert_group_metrics.json"  # 统一格式输出（原始 raw 已备份）

METRIC_ORDER = ["n", "hit@1", "hit@3", "hit@10", "ndcg@10"]

def read_jsonl(p: Path) -> List[Dict[str, Any]]:
    out = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def hit_at_k(ranked: List[int], gold: List[int], k: int) -> float:
    return 1.0 if any(x in gold for x in ranked[:k]) else 0.0

def ndcg_at_k(ranked: List[int], gold: List[int], k: int) -> float:
    dcg = 0.0
    for rank, doc_id in enumerate(ranked[:k], start=1):
        rel = 1.0 if doc_id in gold else 0.0
        if rel > 0:
            dcg += rel / math.log2(rank + 1)
    ideal = [1.0] * min(len(gold), k)
    idcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0

def ensure(acc: Dict[Tuple[str,str], Dict[str, float]], key: Tuple[str,str]):
    if key not in acc:
        acc[key] = {"n": 0, "hit@1": 0.0, "hit@3": 0.0, "hit@10": 0.0, "ndcg@10": 0.0}

def main():
    assert DETAILS.exists(), f"missing {DETAILS}"
    rows = read_jsonl(DETAILS)

    acc: Dict[Tuple[str,str], Dict[str, float]] = {}
    bad = 0

    for r in rows:
        # 你的 details 结构：query_id(str), discipline(str), type(str), gold(list[int]), pred(list[int])
        qid = r.get("query_id")
        gold = r.get("gold")
        pred = r.get("pred")
        disc = r.get("discipline")
        qtype = r.get("type")

        if qid is None or not isinstance(gold, list) or not isinstance(pred, list) or not pred:
            bad += 1
            continue

        # 强制 int
        gold_ids = [int(x) for x in gold]
        ranked = [int(x) for x in pred]

        h1 = hit_at_k(ranked, gold_ids, 1)
        h3 = hit_at_k(ranked, gold_ids, 3)
        h10 = hit_at_k(ranked, gold_ids, 10)
        n10 = ndcg_at_k(ranked, gold_ids, 10)

        keys = [
            ("ALL", "ALL"),
            ("discipline", str(disc) if disc is not None else "UNKNOWN"),
            ("type", str(qtype) if qtype is not None else "UNKNOWN"),
        ]
        for key in keys:
            ensure(acc, key)
            acc[key]["n"] += 1
            acc[key]["hit@1"] += h1
            acc[key]["hit@3"] += h3
            acc[key]["hit@10"] += h10
            acc[key]["ndcg@10"] += n10

    out: Dict[str, Dict[str, float]] = {}
    for (dim, name), m in acc.items():
        n = int(m["n"])
        if n <= 0:
            continue
        out[str((dim, name))] = {
            "n": n,
            "hit@1": m["hit@1"] / n,
            "hit@3": m["hit@3"] / n,
            "hit@10": m["hit@10"] / n,
            "ndcg@10": m["ndcg@10"] / n,
        }

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[OK] wrote normalized:", OUT)
    print("[INFO] groups:", len(out), "| bad_lines:", bad)
    print("[INFO] ALL:", out.get(str(("ALL","ALL"))))

if __name__ == "__main__":
    main()
