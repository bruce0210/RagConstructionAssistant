import json, math, os
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
DEV_Q = DATA_DIR / "dev_queries.jsonl"
DEV_L = DATA_DIR / "dev_labels.json"

META = Path("/home/RagConstructionAssistant/data/index/meta.jsonl")
INDEX = Path("/home/RagConstructionAssistant/data/index/faiss.index")

MODEL = "BAAI/bge-m3"
TOPK = 10

OUT = DATA_DIR / "dev_dense_details.jsonl"
OUT_SUM = DATA_DIR / "dev_dense_group_metrics.json"

def build_offsets(meta_path: Path):
    offsets=[]
    with meta_path.open("rb") as f:
        while True:
            pos=f.tell()
            line=f.readline()
            if not line: break
            offsets.append(pos)
    return offsets

def load_meta_by_faiss_idx(fh, offsets, idx: int):
    fh.seek(offsets[idx])
    line = fh.readline()
    if not line: return None
    return json.loads(line.decode("utf-8", errors="ignore"))

def ndcg_at_k(retrieved_ids, gold_set, k=10):
    rel = [1 if rid in gold_set else 0 for rid in retrieved_ids[:k]]
    dcg = sum(r / math.log2(i+2) for i, r in enumerate(rel))
    ideal_len = min(k, len(gold_set))
    if ideal_len == 0:
        return None
    idcg = sum(1 / math.log2(i+2) for i in range(ideal_len))
    return dcg / idcg if idcg > 0 else 0.0

def main():
    # load queries
    queries=[]
    with DEV_Q.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))

    # load labels
    with DEV_L.open("r", encoding="utf-8") as f:
        labels=json.load(f)

    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", "-1") else "cpu"
    embedder = SentenceTransformer(MODEL, device=device)

    index = faiss.read_index(str(INDEX))
    if hasattr(index, "nprobe"):
        index.nprobe = 16

    offsets = build_offsets(META)

    # group stats containers
    group = {}  # key -> dict counters

    def upd_group(key, hit1, hit3, hit10, ndcg):
        g = group.setdefault(key, {"n":0, "hit1":0, "hit3":0, "hit10":0, "ndcg_sum":0.0, "ndcg_n":0})
        g["n"] += 1
        g["hit1"] += int(hit1)
        g["hit3"] += int(hit3)
        g["hit10"] += int(hit10)
        if ndcg is not None:
            g["ndcg_sum"] += float(ndcg)
            g["ndcg_n"] += 1

    with META.open("rb") as fh, OUT.open("w", encoding="utf-8") as out:
        for q in queries:
            qid = str(q["id"])
            qtext = q["query"]
            disc = q.get("discipline","")
            qtype = q.get("type","")

            gold_ids = labels.get(qid, [])
            gold_set = set(gold_ids)

            vec = embedder.encode([qtext], normalize_embeddings=True)
            vec = np.asarray(vec, dtype="float32")
            D, I = index.search(vec, TOPK)

            top = []
            top_ids = []
            for rank, idx in enumerate(I[0], start=1):
                rec = load_meta_by_faiss_idx(fh, offsets, int(idx))
                clause_id = rec.get("id")
                score = float(D[0][rank-1])
                top.append({
                    "rank": rank,
                    "clause_id": clause_id,
                    "score": score,
                    "source": rec.get("source"),
                    "clause_no": rec.get("clause_no"),
                    "clause": rec.get("clause"),
                    "text": (rec.get("text") or "")[:220].replace("\n"," ")
                })
                top_ids.append(clause_id)

            # metrics per query
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

    # finalize group metrics
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

    with OUT_SUM.open("w", encoding="utf-8") as f:
        json.dump(finalize(group), f, ensure_ascii=False, indent=2)

    print("[OK] wrote:", OUT)
    print("[OK] wrote:", OUT_SUM)

if __name__ == "__main__":
    main()
