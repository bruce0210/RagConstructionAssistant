import os, json, math, argparse
from collections import defaultdict

from colbert import Searcher
from colbert.infra import Run, RunConfig

def pick(d, keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def load_labels(path):
    obj = json.load(open(path, "r", encoding="utf-8"))
    # 支持：dict[qid] -> list/str  或 list(与 queries 同序)
    return obj

def ensure_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]

def metrics_at_k(pred, gold, k):
    pred_k = pred[:k]
    gold_set = set(gold)

    hit_positions = [i for i, cid in enumerate(pred_k) if cid in gold_set]
    hit = 1.0 if hit_positions else 0.0

    # MRR
    if hit_positions:
        rr = 1.0 / (hit_positions[0] + 1)
    else:
        rr = 0.0

    # nDCG (binary relevance)
    dcg = 0.0
    for i, cid in enumerate(pred_k):
        rel = 1.0 if cid in gold_set else 0.0
        if rel > 0:
            dcg += rel / math.log2(i + 2)
    idcg = 0.0
    for i in range(min(len(gold_set), k)):
        idcg += 1.0 / math.log2(i + 2)
    ndcg = (dcg / idcg) if idcg > 0 else 0.0

    # Recall@k（命中任一即算 1；若你 gold>1 想按覆盖率，可自行改）
    recall = hit

    return {"hit@k": hit, "mrr@k": rr, "ndcg@k": ndcg, "recall@k": recall}

def agg_metrics(rows, k):
    if not rows:
        return {"n": 0, "hit@k": 0.0, "mrr@k": 0.0, "ndcg@k": 0.0, "recall@k": 0.0}
    s = defaultdict(float)
    for r in rows:
        m = r["metrics"][str(k)]
        for kk, vv in m.items():
            s[kk] += float(vv)
    n = len(rows)
    out = {"n": n}
    for kk in ["hit@k", "mrr@k", "ndcg@k", "recall@k"]:
        out[kk] = s[kk] / n
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="data/dev_queries.jsonl")
    ap.add_argument("--labels", default="data/dev_labels.json")
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--topk", type=int, default=20)

    # 这三个与 build 时保持一致
    ap.add_argument("--root", default="/home/RagConstructionAssistant/experiments/retrieval/experiments")
    ap.add_argument("--experiment", default="default")
    ap.add_argument("--index_name", default="colbert_jina_cpu_full")

    ap.add_argument("--pid2clause", default="/home/RagConstructionAssistant/data/index/colbert/colbert_jina_cpu_full_pid2clause.json")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_details = os.path.join(args.out_dir, "dev_colbert_details.jsonl")
    out_group = os.path.join(args.out_dir, "dev_colbert_group_metrics.json")

    queries = load_jsonl(args.queries)
    labels = load_labels(args.labels)
    pid2clause = json.load(open(args.pid2clause, "r", encoding="utf-8"))

    # 建一个 qid -> gold 映射（尽量兼容）
    def get_gold(qobj, idx):
        qid = pick(qobj, ["query_id", "qid", "id", "question_id"], None)
        if isinstance(labels, dict):
            if qid is None:
                # fallback：用 idx 当 key
                return ensure_list(labels.get(str(idx), []))
            return ensure_list(labels.get(str(qid), labels.get(qid, [])))
        if isinstance(labels, list):
            # 同序
            if idx < len(labels):
                return ensure_list(labels[idx])
        return []

    rows = []
    with Run().context(RunConfig(nranks=1, root=args.root, experiment=args.experiment)):
        searcher = Searcher(index=args.index_name)

        with open(out_details, "w", encoding="utf-8") as w:
            for i, q in enumerate(queries):
                qtext = pick(q, ["query", "question", "text"], "")
                discipline = pick(q, ["discipline", "domain", "major"], "UNKNOWN")
                qtype = pick(q, ["type", "qtype", "question_type"], "UNKNOWN")
                qid = pick(q, ["query_id", "qid", "id", "question_id"], str(i))

                gold = get_gold(q, i)
                # ColBERT 返回 pids/ranks/scores
                pids, ranks, scores = searcher.search(qtext, k=args.topk)
                # pids 可能是 list[int] 或 ndarray；统一转 int
                pred_clause = []
                for pid in list(pids):
                    pid_int = int(pid)
                    if 0 <= pid_int < len(pid2clause):
                        pred_clause.append(pid2clause[pid_int])

                per_k = {}
                for k in [1, 5, 10, args.topk]:
                    per_k[str(k)] = metrics_at_k(pred_clause, gold, k)

                row = {
                    "query_id": str(qid),
                    "query": qtext,
                    "discipline": discipline,
                    "type": qtype,
                    "gold": gold,
                    "pred": pred_clause,
                    "metrics": per_k,
                }
                w.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append(row)

    # group metrics：overall + by discipline + by type
    ks = [1, 5, 10, args.topk]
    out = {"overall": {}, "by_discipline": {}, "by_type": {}, "topk": args.topk}

    for k in ks:
        out["overall"][str(k)] = agg_metrics(rows, k)

    by_disc = defaultdict(list)
    by_type = defaultdict(list)
    for r in rows:
        by_disc[r["discipline"]].append(r)
        by_type[r["type"]].append(r)

    for d, rs in by_disc.items():
        out["by_discipline"][d] = {str(k): agg_metrics(rs, k) for k in ks}
    for t, rs in by_type.items():
        out["by_type"][t] = {str(k): agg_metrics(rs, k) for k in ks}

    with open(out_group, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote: {out_details}")
    print(f"[OK] wrote: {out_group}")

if __name__ == "__main__":
    main()
