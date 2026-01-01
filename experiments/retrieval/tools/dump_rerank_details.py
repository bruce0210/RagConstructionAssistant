import json, math, os
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

DATA_DIR = Path("data")
DEV_Q = DATA_DIR / "dev_queries.jsonl"
DEV_L = DATA_DIR / "dev_labels.json"

META = Path("/home/RagConstructionAssistant/data/index/meta.jsonl")
INDEX = Path("/home/RagConstructionAssistant/data/index/faiss.index")
DENSE_MODEL = "BAAI/bge-m3"

TOPK = 10
DENSE_K = 200
BM25_K = 200
RRF_K = 60
RERANK_N = 50

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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="BAAI/bge-reranker-base")
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    # meta
    metas=[]
    corpus_tokens=[]
    id2meta={}
    with META.open("r", encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            metas.append(m)
            corpus_tokens.append(tokenize(m.get("text","")))
            cid = as_int(m.get("id"))
            id2meta[cid] = m

    bm25 = BM25Okapi(corpus_tokens)

    # queries
    queries=[]
    with DEV_Q.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))

    # labels
    with DEV_L.open("r", encoding="utf-8") as f:
        labels=json.load(f)

    # dense
    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", "-1") else "cpu"
    embedder = SentenceTransformer(DENSE_MODEL, device=device)
    index = faiss.read_index(str(INDEX))
    if hasattr(index, "nprobe"):
        index.nprobe = 16

    # reranker
    reranker = CrossEncoder(args.model, device=device)

    out_details = DATA_DIR / "dev_rerank_details.jsonl"
    out_group   = DATA_DIR / "dev_rerank_group_metrics.json"

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

            gold_ids = [as_int(x) for x in labels.get(qid, [])]
            gold_set = set(gold_ids)

            # --- hybrid candidates via RRF ---
            vec = embedder.encode([qtext], normalize_embeddings=True)
            vec = np.asarray(vec, dtype="float32")
            D, I = index.search(vec, DENSE_K)
            dense_rank = {}
            for r, idx in enumerate(I[0], start=1):
                if idx < 0: 
                    continue
                cid = as_int(metas[int(idx)].get("id"))
                dense_rank[cid] = r

            scores = bm25.get_scores(tokenize(qtext))
            top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:BM25_K]
            bm25_rank = {}
            for r, i in enumerate(top_idx, start=1):
                cid = as_int(metas[i].get("id"))
                bm25_rank[cid] = r

            cand = set(dense_rank) | set(bm25_rank)
            merged = []
            for cid in cand:
                s = 0.0
                if cid in dense_rank:
                    s += 1.0 / (RRF_K + dense_rank[cid])
                if cid in bm25_rank:
                    s += 1.0 / (RRF_K + bm25_rank[cid])
                merged.append((cid, s))
            merged.sort(key=lambda x: x[1], reverse=True)

            cids = [cid for cid,_ in merged[:RERANK_N]]

            # --- rerank ---
            pairs = [(qtext, (id2meta.get(cid, {}).get("text") or "")) for cid in cids]
            r_scores = reranker.predict(pairs, batch_size=args.batch_size)

            reranked = sorted(zip(cids, r_scores), key=lambda x: float(x[1]), reverse=True)

            top=[]
            top_ids=[]
            for rank, (cid, sc) in enumerate(reranked[:TOPK], start=1):
                m = id2meta.get(cid, {})
                top.append({
                    "rank": rank,
                    "clause_id": cid,
                    "score": float(sc),
                    "source": m.get("source"),
                    "clause_no": m.get("clause_no"),
                    "clause": m.get("clause"),
                    "text": (m.get("text") or "")[:220].replace("\n"," ")
                })
                top_ids.append(cid)

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
