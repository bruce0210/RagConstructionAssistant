# search.py
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ---- 路径：和 Home.py 一致 ----
REPO_ROOT = Path(__file__).resolve().parents[2]   # .../RagConstructionAssistant
INDEX_DIR = REPO_ROOT / "data" / "index"

META_PATH = INDEX_DIR / "meta.jsonl"
INDEX_PATH = INDEX_DIR / "faiss.index"

# ---- 1. 载入 meta.jsonl ----
metas = []
with META_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        metas.append(json.loads(line))

print(f"Loaded {len(metas)} meta rows from {META_PATH}")

# ---- 2. 载入 FAISS index ----
index = faiss.read_index(str(INDEX_PATH))
print(f"FAISS ntotal={index.ntotal}, dim={index.d}")

# ---- 3. 加载与建库一致的 embedder（BAAI/bge-m3） ----
MODEL_NAME = "BAAI/bge-m3"

# 和 Home.py / ingest_docx.get_embedder 的行为保持一致：normalize_embeddings=True
device = "cuda" if (not np.allclose(0, 0) and  # 小 trick 防 linters
                    (Path("/dev/nvidia0").exists() or Path("/dev/nvidiactl").exists())) else "cpu"
embedder = SentenceTransformer(MODEL_NAME, device=device)

def encode_query(text: str) -> np.ndarray:
    """
    关键：不要加 'query:' 前缀，直接用原始中文问题，
    并且 normalize_embeddings=True，dtype=float32。
    """
    emb = embedder.encode([text], normalize_embeddings=True)
    return emb.astype("float32")

def search(query: str, top_k: int = 10):
    q_vec = encode_query(query)
    D, I = index.search(q_vec, top_k)   # inner product on normalized vectors ≈ cos-sim

    scores = D[0]
    idxs   = I[0]

    results = []
    for rank, (idx, score) in enumerate(zip(idxs, scores), start=1):
        if idx < 0 or idx >= len(metas):
            continue
        m = metas[int(idx)]
        results.append({
            "rank": rank,
            "score": float(score),             # 0~1，后续可 *100 变百分比
            "id": m.get("id", idx),
            "source": m.get("source", ""),
            "clause_no": m.get("clause_no", ""),
            "clause": m.get("clause", ""),
            "text": (m.get("text") or "").strip(),
            "media": m.get("media") or [],
        })
    return results

if __name__ == "__main__":
    while True:
        q = input("\n请输入规范问题（直接回车退出）：").strip()
        if not q:
            break
        hits = search(q, top_k=5)
        print("=" * 80)
        print("Query:", q)
        if not hits:
            print("😢 未检索到结果")
            continue
        for r in hits:
            print(f"\n[Top {r['rank']}]  score={r['score']*100:.2f}%")
            print(f"  来源: {r['source']}")
            print(f"  条号: {r['clause_no']} / {r['clause']}")
            print("  内容:", r["text"][:200].replace("\n", " ") + "...")
