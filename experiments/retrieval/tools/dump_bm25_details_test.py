import json, math
from pathlib import Path

from rank_bm25 import BM25Okapi

DATA_DIR = Path("data")
DEV_Q = DATA_DIR / "test_queries.jsonl"
DEV_L = DATA_DIR / "test_labels.json"

META = Path("/home/RagConstructionAssistant/data/index/meta.jsonl")

TOPK = 10
CAND_K = 200  # BM25先取Top200，再截Top10写明细

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

def as_int(x):
    if isinstance(x, int):
        return x
    if isinstance(x, str) and x.isdigit():
        return int(x)
    return x

def ndcg_at_k(retrieved_ids, gold_set, k=10):
    rel = [1 if rid in gold_set else 0 for rid in retrieved_ids[:k]]
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
    ideal_len = min(k, len(gold_set))
    if ideal_len == 0:
        return None
    idcg = sum(1 / math.log2(i + 2) for i in range(ideal_len))
    return dcg / idcg if idcg > 0 else 0.0

def main():
    # load meta
    metas = []
    corpus_tokens = []
    with META.open("r", encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            metas.append(m)
            corpus_tokens.append(tokenize(m.get("text","")))

    bm25 = BM25Okapi(corpus_tokens)

    # load dev queries
    queries=[]
    with DEV_Q.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))

    # load labels
    with DEV_L.open("r", encoding="utf-8") as f:
        labels=json.load(f)

    out_details = DATA_DIR / "test_bm25_details.jsonl"
    out_group   = DATA_DIR / "test_bm25_group_metrics.json"

    group = {}
    def upd_group(key, hit1, hit3, hit10, ndcg):
        g = group.setdefault(key, {"n":0, "hit1":0, "hit3":0, "hit10":0, "ndcg_sum":0.0, "ndcg_n":0})
        g["n"] += 1
        g["hit1"] += int(hit1)
        g["hit3"] += int(hit3)
        g["hit10"] += int(hit10)
        if ndcg is not None:
            g["ndcg_sum"] += float(ndcg)
            g["ndcg_n"] += 1

    with out_details.open("w", encoding="utf-8") as out:
        for q in queries:
            qid = str(q["id"])
            qtext = q["query"]
            disc = q.get("discipline","")
            qtype = q.get("type","")

            gold_ids = labels.get(qid, [])
            gold_ids = [as_int(x) for x in gold_ids]
            gold_set = set(gold_ids)

            scores = bm25.get_scores(tokenize(qtext))
            top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:CAND_K]

            top = []
            top_ids = []
            for rank, i in enumerate(top_idx[:TOPK], start=1):
                rec = metas[i]
                clause_id = as_int(rec.get("id"))
                top.append({
                    "rank": rank,
                    "clause_id": clause_id,
                    "score": float(scores[i]),
                    "source": rec.get("source"),
                    "clause_no": rec.get("clause_no"),
                    "clause": rec.get("clause"),
                    "text": (rec.get("text") or "")[:220].replace("\n"," ")
                })
                top_ids.append(clause_id)

            hit1 = (len(top_ids) >= 1 and top_ids[0] in gold_set)
            hit3 = any(rid in gold_set for rid in top_ids[:3])
            hit10 = any(rid in gold_set for rid in top_ids[:10])
            first_hit_rank = None
            for i, rid in enumerate(top_ids[:10], start=1):
                if rid in gold_set:
                    first_hit_rank = i
                    break
            ndcg = ndcg_at_k(top_ids, gold_set, k=10)

            upd_group(("ALL","ALL"), hit1, hit3, hit10, ndcg)
            upd_group(("discipline", disc), hit1, hit3, hit10, ndcg)
            upd_group(("type", qtype), hit1, hit3, hit10, ndcg)

            out.write(json.dumps({
                "id": qid,
                "discipline": disc,
                "type": qtype,
                "query": qtext,
                "gold_ids": gold_ids,
                "top10": top,
                "hit1": hit1,
                "hit3": hit3,
                "hit10": hit10,
                "first_hit_rank": first_hit_rank,
                "ndcg10": ndcg
            }, ensure_ascii=False) + "\n")

    def finalize(d):
        out = {}
        for k,v in d.items():
            n=v["n"]
            out[str(k)] = {
                "n": n,
                "hit@1": v["hit1"]/n if n else None,
                "hit@3": v["hit3"]/n if n else None,
                "hit@10": v["hit10"]/n if n else None,
                "ndcg@10": (v["ndcg_sum"]/v["ndcg_n"]) if v["ndcg_n"] else None
            }
        return out

    with out_group.open("w", encoding="utf-8") as f:
        json.dump(finalize(group), f, ensure_ascii=False, indent=2)

    print("[OK] wrote:", out_details)
    print("[OK] wrote:", out_group)

if __name__ == "__main__":
    main()
