# interface/pages/1_File_Upload.py
from __future__ import annotations
import os, time, json
from pathlib import Path
from typing import List, Dict, Any
import streamlit as st
import numpy as np

from core.retrieval.ingest_docx import (
    REPO_ROOT, INDEX_DIR, MEDIA_DIR,
    parse_docx_into_clauses, get_embedder, embed_texts, build_faiss_index
)
from core.utils.oss_io import get_oss_clients, oss_put

st.set_page_config(page_title="Batch DOCX Ingest to OSS", page_icon="📄", layout="wide")
st.title("📄 批量上传 DOCX → 解析/建索引 → 分桶上 OSS")
st.caption("原始 DOCX 存 A 桶；抽取的图片存 B 桶；索引与元数据可选存 C 桶。ECS 内网上传，前端用公网 URL 访问。")

# ---- 参数区
with st.sidebar:
    model_name = st.selectbox(
        "Embedding 模型",
        ["BAAI/bge-base-zh-v1.5", "BAAI/bge-m3"],
        index=0
    )
    batch = st.number_input("Embed batch size", 8, 1024, 64, 8)
    force_cpu = st.toggle("FAISS 仅用 CPU", value=False)
    backup_idx = st.toggle("把 faiss.index / meta.jsonl 备份到 OSS", value=True)

# ---- 读取 OSS 客户端
try:
    secrets = st.secrets if "oss" in st.secrets else {}
    bucket_docx, bucket_media, bucket_index, url_docx, url_media, url_index = get_oss_clients(secrets)
    st.success("✅ OSS 已就绪（内网上传 + 公网访问）。")
except Exception as e:
    st.error(f"❌ OSS 配置错误：{e}")
    st.stop()

# ---- 文件上传
files = st.file_uploader("选择一个或多个 DOCX", type=["docx"], accept_multiple_files=True)
if not files:
    st.info("请先选择 DOCX。")
    st.stop()

# ---- 临时落地
TMP = REPO_ROOT / ".tmp_uploads"
TMP.mkdir(parents=True, exist_ok=True)
local_paths: List[Path] = []
for f in files:
    stamped = f"{int(time.time())}_{Path(f.name).name}"
    p = TMP / stamped
    with open(p, "wb") as out:
        out.write(f.read())
    local_paths.append(p)

st.write(f"📦 已接收 {len(local_paths)} 个文件。")

# ---- 解析 & 抽图（图片会落在本地 MEDIA_DIR）
all_clauses: List[Dict[str, Any]] = []
for i, p in enumerate(local_paths, 1):
    st.write(f"🔍 解析：{p.name}")
    cs = parse_docx_into_clauses(p)
    st.write(f"　└─ {len(cs)} 条款")
    all_clauses.extend(cs)

if not all_clauses:
    st.warning("未解析到条款。")
    st.stop()

# ---- 上传 DOCX 到 A 桶（并生成公网 URL）
st.subheader("上传 DOCX 到 OSS（A 桶）")
docx_url_map: Dict[str, str] = {}
for p in local_paths:
    key = f"docx/{p.name}"
    oss_put(bucket_docx, p, key)
    u = url_docx(key)
    docx_url_map[p.name] = u
    st.write(f"☁️ {p.name} → {u}")

# ---- 上传图片到 B 桶，并把 meta 里的 media 改为公网 URL
st.subheader("上传图片到 OSS（B 桶），回填 URL")
uploaded_media: Dict[str, str] = {}
for c in all_clauses:
    new_media = []
    for rel in c.get("media", []):
        if rel in uploaded_media:
            new_media.append(uploaded_media[rel])
            continue
        abs_path = (REPO_ROOT / rel).resolve()
        if not abs_path.exists():
            continue
        # 以 MEDIA_DIR 的相对路径组织 key
        try:
            rel_key = abs_path.relative_to(MEDIA_DIR).as_posix()
        except Exception:
            rel_key = abs_path.name
        key = f"media/{rel_key}"
        oss_put(bucket_media, abs_path, key)
        u = url_media(key)
        uploaded_media[rel] = u
        new_media.append(u)
    c["media"] = new_media

# ---- 生成向量 & 构建索引（本地）
st.subheader("Embedding & FAISS")
os.environ["RAG_EMBED_MODEL"] = model_name
model = get_embedder(model_name)
texts = [c["text"] for c in all_clauses]
vecs = embed_texts(model, texts, batch_size=int(batch))
index = build_faiss_index(vecs, use_gpu_if_possible=not force_cpu)

INDEX_DIR.mkdir(parents=True, exist_ok=True)
faiss_path = INDEX_DIR / "faiss.index"
meta_path  = INDEX_DIR / "meta.jsonl"

import faiss
faiss.write_index(index, str(faiss_path))

# meta：补充 source_url（DOCX 的公网 URL）
with open(meta_path, "w", encoding="utf-8") as f:
    for c in all_clauses:
        src = c.get("source", "")
        # 尝试匹配时间戳文件名
        hit = None
        for p in local_paths:
            if src == p.name or src == p.name.split("_", 1)[-1]:
                hit = docx_url_map.get(p.name)
                break
        if hit:
            c["source_url"] = hit
        f.write(json.dumps(c, ensure_ascii=False) + "\n")

st.success(f"✅ 本地索引：{faiss_path}")
st.success(f"✅ 本地元数据：{meta_path}")

# ---- 可选：把索引与 meta 也备份到 OSS（C 桶）
if backup_idx and bucket_index is not None:
    idx_key, meta_key = "index/faiss.index", "index/meta.jsonl"
    oss_put(bucket_index, faiss_path, idx_key)
    oss_put(bucket_index, meta_path,  meta_key)
    st.write(f"☁️ 索引 → {url_index(idx_key)}")
    st.write(f"☁️ 元数据 → {url_index(meta_key)}")

st.balloons()
st.success("🎉 完成：DOCX 入 A 桶、图片入 B 桶、索引已建（C 桶备份可选）。")
