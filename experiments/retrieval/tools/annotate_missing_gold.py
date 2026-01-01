import argparse, json, os
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def build_offsets(meta_path: Path):
    offsets = []
    with meta_path.open("rb") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            offsets.append(pos)
    return offsets


def load_record_by_idx(fh, offsets, idx: int):
    fh.seek(offsets[idx])
    line = fh.readline()
    if not line:
        return None
    return json.loads(line.decode("utf-8", errors="ignore"))


def pretty_print(rec: dict, rank: int, score: float):
    clause_id = rec.get("id")
    source = rec.get("source")
    clause_no = rec.get("clause_no")
    clause = rec.get("clause")
    text = (rec.get("text") or "").replace("\n", " ")
    # cosine similarity if normalized embeddings + IP index; percent just for readability
    pct = score * 100.0
    print(f"\n[Top {rank:02d}]  score={pct:.2f}%   clause_id={clause_id}")
    print(f"  source   : {source}")
    print(f"  clause_no: {clause_no} / clause: {clause}")
    print(f"  text     : {text[:260]}")


def parse_id_list(s: str):
    # accept "1,3,5" / "1 3 5" / "1，3，5"
    s = s.strip().replace("，", ",")
    if not s:
        return []
    parts = []
    for chunk in s.replace(" ", ",").split(","):
        chunk = chunk.strip()
        if chunk:
            parts.append(int(chunk))
    return parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--missing", required=True, help="missing_queries.jsonl")
    ap.add_argument("--labels", required=True, help="dev_labels.json")
    ap.add_argument("--meta", required=True, help="meta.jsonl")
    ap.add_argument("--index", required=True, help="faiss.index")
    ap.add_argument("--model", default="BAAI/bge-m3", help="embedding model name or local path")
    ap.add_argument("--topk", type=int, default=200, help="retrieve topK from FAISS")
    ap.add_argument("--show", type=int, default=20, help="print topN results")
    ap.add_argument("--nprobe", type=int, default=16, help="IVF nprobe if applicable")
    args = ap.parse_args()

    missing_path = Path(args.missing)
    labels_path  = Path(args.labels)
    meta_path    = Path(args.meta)
    index_path   = Path(args.index)

    if not missing_path.exists():
        print(f"[ERR] missing file not found: {missing_path}")
        return
    if not labels_path.exists():
        print(f"[ERR] labels file not found: {labels_path}")
        return
    if not meta_path.exists():
        print(f"[ERR] meta file not found: {meta_path}")
        return
    if not index_path.exists():
        print(f"[ERR] index file not found: {index_path}")
        return

    # load labels
    with labels_path.open("r", encoding="utf-8") as f:
        labels = json.load(f)

    # load missing queries
    missing = []
    with missing_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                missing.append(json.loads(line))

    print(f"[INFO] missing queries: {len(missing)}")
    print(f"[INFO] labels keys before: {len(labels)}")

    # load model + index
    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", "-1") else "cpu"
    print(f"[INFO] loading model: {args.model}  device={device}")
    embedder = SentenceTransformer(args.model, device=device)

    print(f"[INFO] loading faiss index: {index_path}")
    index = faiss.read_index(str(index_path))
    if hasattr(index, "nprobe"):
        index.nprobe = int(args.nprobe)

    # offsets for random access
    print(f"[INFO] building meta offsets (one-time scan): {meta_path}")
    offsets = build_offsets(meta_path)
    print(f"[INFO] meta rows = {len(offsets)} ; faiss ntotal = {index.ntotal}")

    # open meta file handle once
    with meta_path.open("rb") as fh:
        for q in missing:
            qid = str(q["id"])
            discipline = q.get("discipline","")
            qtype = q.get("type","")
            query = q.get("query","")

            if qid in labels:
                # already filled, skip
                continue

            print("\n" + "="*100)
            print(f"[QID={qid}]  {discipline} / {qtype}")
            print(f"Original query: {query}")

            while True:
                helper = input("Helper query (回车=用原句；输入 /skip 跳过此题)： ").strip()
                if helper == "/skip":
                    break
                use_q = helper if helper else query

                vec = embedder.encode([use_q], normalize_embeddings=True)
                vec = np.asarray(vec, dtype="float32")
                D, I = index.search(vec, args.topk)

                # print topN
                shown = 0
                print("\n----- Candidates (show top {}) -----".format(args.show))
                for rank, idx in enumerate(I[0], start=1):
                    if idx < 0:
                        continue
                    rec = load_record_by_idx(fh, offsets, int(idx))
                    if rec is None:
                        continue
                    pretty_print(rec, rank, float(D[0][rank-1]))
                    shown += 1
                    if shown >= args.show:
                        break

                print("\n输入 gold clause_id（逗号分隔，如 70518,92962；输入 /retry 重新换问法；回车=再换问法）：")
                ans = input("gold ids = ").strip()
                if ans == "/retry" or ans == "":
                    continue

                gold_ids = parse_id_list(ans)
                labels[qid] = gold_ids

                # write back immediately (safe overwrite)
                tmp = labels_path.with_suffix(".json.tmp")
                with tmp.open("w", encoding="utf-8") as f:
                    json.dump(labels, f, ensure_ascii=False, indent=2)
                tmp.replace(labels_path)

                print(f"[OK] saved qid={qid} gold={gold_ids}  (labels now {len(labels)})")
                break

    print(f"\n[DONE] labels keys after: {len(labels)}")
    print(f"labels file: {labels_path}")


if __name__ == "__main__":
    main()
