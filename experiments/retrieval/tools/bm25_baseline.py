# tools/bm25_baseline.py
import argparse, json, math, os
from collections import defaultdict

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)

def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def has_cjk(s: str) -> bool:
    for ch in s:
        o = ord(ch)
        if 0x4E00 <= o <= 0x9FFF:
            return True
    return False

def tokenize(text: str):
    text = (text or "").strip()
    if not text:
        return []
    if has_cjk(text):
        import jieba
        return [t for t in jieba.lcut(text) if t.strip()]
    return [t for t in text.split() if t.strip()]

def pick(d, keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

def ensure_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]

def hit_at_k(pred_ids, gold_ids, k):
    s = set(gold_ids)
    return 1.0 if any(pid in s for pid in pred_ids[:k]) else 0.0

def ndcg_at_k(pred_ids, gold_ids, k):
    gold = set(gold_ids)
    dcg = 0.0
    for i, pid in enumerate(pred_ids[:k]):
        rel = 1.0 if pid in gold else 0.0
        if rel > 0:
            dcg += (2**rel - 1.0) / math.log2(i + 2)
    # IDCG: all relevant ranked at top
    m = min(len(gold), k)
    idcg = 0.0
    for i in range(m):
        idcg += (2**1.0 - 1.0) / math.log2(i + 2)
    return dcg / idcg if idcg > 0 else 0.0

def metrics_for_query(pred_ids, gold_ids):
    return {
        "hit@1": hit_at_k(pred_ids, gold_ids, 1),
        "hit@3": hit_at_k(pred_ids, gold_ids, 3),
        "hit@10": hit_at_k(pred_ids, gold_ids, 10),
        "ndcg@10": ndcg_at_k(pred_ids, gold_ids, 10),
    }

def avg_metrics(ms):
    if not ms:
        return {"hit@1":0,"hit@3":0,"hit@10":0,"ndcg@10":0,"n":0}
    out = {k:0.0 for k in ["hit@1","hit@3","hit@10","ndcg@10"]}
    for m in ms:
        for k in out:
            out[k] += float(m.get(k, 0.0))
    for k in out:
        out[k] /= len(ms)
    out["n"] = len(ms)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--out_dir", default="artifacts/bm25")
    args = ap.parse_args()

    # 1) load meta
    doc_ids, doc_texts, corpus_tokens = [], {}, []
    for row in read_jsonl(args.meta):
        did = str(pick(row, ["clause_id","docid","id","clauseId","clauseID"]))
        txt = pick(row, ["text","content","clause_text","clauseText","body"], "")
        doc_ids.append(did)
        doc_texts[did] = txt
        corpus_tokens.append(tokenize(txt))

    # 2) build BM25
    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi(corpus_tokens)

    # 3) run + eval
    details_rows = []
    overall_ms = []
    group_ms = {
        "discipline": defaultdict(list),
        "type": defaultdict(list),
    }

    for q in read_jsonl(args.dev):
        qid = str(pick(q, ["qid","query_id","id","question_id","questionId"]))
        qtext = pick(q, ["query","question","text","query_text","question_text"], "")
        gold = ensure_list(pick(q, ["gold_clause_ids","gold","answers","answer_clause_ids","clause_ids","labels","label"], []))
        gold = [str(x) for x in gold]

        scores = bm25.get_scores(tokenize(qtext))
        # topk indices
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:args.topk]
        hits = []
        pred_ids = []
        for r, i in enumerate(top_idx, start=1):
            did = doc_ids[i]
            pred_ids.append(did)
            hits.append({
                "rank": r,
                "clause_id": did,
                "score": float(scores[i]),
                "text_preview": (doc_texts.get(did,"")[:180] + "…") if len(doc_texts.get(did,"")) > 180 else doc_texts.get(did,""),
            })

        m = metrics_for_query(pred_ids, gold)
        overall_ms.append(m)

        disc = str(pick(q, ["discipline","major","domain"], "UNKNOWN"))
        typ  = str(pick(q, ["type","question_type","qtype"], "UNKNOWN"))
        group_ms["discipline"][disc].append(m)
        group_ms["type"][typ].append(m)

        details_rows.append({
            "qid": qid,
            "query": qtext,
            "gold_clause_ids": gold,
            "pred_clause_ids": pred_ids[:10],
            "metrics": m,
            "hits": hits,
        })

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    details_path = os.path.join(out_dir, "dev_bm25_details.jsonl")
    write_jsonl(details_path, details_rows)

    overall = avg_metrics(overall_ms)
    metrics_path = os.path.join(out_dir, "dev_bm25_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"overall": overall}, f, ensure_ascii=False, indent=2)

    group_out = {"overall": overall, "discipline": {}, "type": {}}
    for k in ["discipline","type"]:
        for g, ms in group_ms[k].items():
            group_out[k][g] = avg_metrics(ms)

    group_path = os.path.join(out_dir, "dev_bm25_group_metrics.json")
    with open(group_path, "w", encoding="utf-8") as f:
        json.dump(group_out, f, ensure_ascii=False, indent=2)

    print("OK")
    print("details:", details_path)
    print("metrics:", metrics_path)
    print("group:", group_path)

if __name__ == "__main__":
    main()
