# inspect_index.py
import json
import faiss

META_PATH = "../../data/index/meta.jsonl"
INDEX_PATH = "../../data/index/faiss.index"

# 1) 读取 meta.jsonl
metas = []
with open(META_PATH, "r", encoding="utf-8") as f:
    for line in f:
        metas.append(json.loads(line))

print("Meta 行数:", len(metas))
print("首行示例:", metas[0])
print("末行示例:", metas[-1])

# 检查 id 连续性（可选）
bad = [i for i, m in enumerate(metas) if m.get("id") != i]
print("id 不连续的数量:", len(bad))

# 2) 读取 faiss.index
index = faiss.read_index(INDEX_PATH)
print("FAISS 向量数:", index.ntotal)
print("向量维度:", index.d)
