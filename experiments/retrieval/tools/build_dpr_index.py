# tools/build_dpr_index.py
import json, os
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

META = Path("/home/RagConstructionAssistant/data/index/meta.jsonl")
OUT  = Path("/home/RagConstructionAssistant/data/index/faiss_dpr_e5.index")

MODEL = "intfloat/multilingual-e5-base"
BATCH = 128

def gpu_available():
    return Path("/dev/nvidia0").exists() or Path("/dev/nvidiactl").exists()

def iter_texts(meta_path: Path):
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            txt = (row.get("text") or "").strip()
            yield txt

def main():
    device = "cuda" if gpu_available() else "cpu"
    print(f"[INFO] device={device}")
    print(f"[INFO] model={MODEL}")
    print(f"[INFO] meta ={META}")
    print(f"[INFO] out  ={OUT}")

    embedder = SentenceTransformer(MODEL, device=device)

    buf = []
    index = None
    total = 0

    for txt in tqdm(iter_texts(META), desc="Encoding+Adding"):
        # E5 标准：passage 前缀
        buf.append("passage: " + txt)
        if len(buf) >= BATCH:
            vecs = embedder.encode(buf, normalize_embeddings=True)
            vecs = np.asarray(vecs, dtype="float32")
            if index is None:
                dim = vecs.shape[1]
                # DPR-style：用 inner product + normalize ≈ cosine
                index = faiss.IndexFlatIP(dim)
                print(f"[INFO] FAISS dim={dim}")
            index.add(vecs)
            total += len(buf)
            buf = []

    if buf:
        vecs = embedder.encode(buf, normalize_embeddings=True)
        vecs = np.asarray(vecs, dtype="float32")
        if index is None:
            dim = vecs.shape[1]
            index = faiss.IndexFlatIP(dim)
            print(f"[INFO] FAISS dim={dim}")
        index.add(vecs)
        total += len(buf)

    print(f"[OK] total vectors added: {total}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(OUT))
    print(f"[OK] wrote: {OUT}")

if __name__ == "__main__":
    main()
