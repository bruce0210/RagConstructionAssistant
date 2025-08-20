# interface/pages/1_File_Upload.py
from __future__ import annotations
import os, sys, time, json
from pathlib import Path
from typing import List, Dict, Any
import streamlit as st
import numpy as np

# ---- make project root importable ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# --------------------------------------

# --- fix import path so "core" can be found ---
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
# ----------------------------------------------

# ---- import ingest module in a reload-safe way ----
import importlib
import core.retrieval.ingest_docx as ingest_docx
ingest_docx = importlib.reload(ingest_docx)  # 强制使用最新版本，避免旧缓存
# 从模块对象上取属性，避免“按名导入命中旧版本”的问题
REPO_ROOT  = ingest_docx.REPO_ROOT
INDEX_DIR  = ingest_docx.INDEX_DIR
MEDIA_DIR  = ingest_docx.MEDIA_DIR
parse_docx_into_clauses = ingest_docx.parse_docx_into_clauses
get_embedder            = ingest_docx.get_embedder
embed_texts             = ingest_docx.embed_texts
build_faiss_index       = ingest_docx.build_faiss_index
write_faiss_index       = ingest_docx.write_faiss_index
# ---------------------------------------------------

from core.utils.oss_io import get_oss_clients, oss_put

# Title and welcome message
st.set_page_config(
    page_title="File Upload",
    page_icon="📄",
    layout="centered"
)

st.title("📄 File Upload")
st.caption("Upload construction-related documents & Batch DOCX Ingest to JSON.")

# ---- 参数区
with st.sidebar:
    model_name = st.selectbox(
        "Embedding Model",
        ["BAAI/bge-base-zh-v1.5", "BAAI/bge-m3"],
        index=0
    )
    batch = st.number_input("Embed batch size", 8, 1024, 64, 8)
    force_cpu = st.toggle("FAISS uses CPU only", value=False)
    backup_idx = st.toggle("Back up faiss.index / meta.jsonl to OSS", value=False)
    st.markdown("---")

    st.caption("📄 Upload construction-related documents & Batch DOCX Ingest to JSON.")
    st.markdown(
        """
        <a href="https://github.com/bruce0210/rag_construction_assistant" target="_blank">
            <img src="https://github.com/codespaces/badge.svg" alt="Open in GitHub">
        </a>
        """,
        unsafe_allow_html=True,
    )

# ---- 读取 OSS 客户端
try:
    secrets = st.secrets if "oss" in st.secrets else {}
    bucket_docx, bucket_media, bucket_index, url_docx, url_media, url_index = get_oss_clients(secrets)
    st.success("✅OSS is ready (intranet upload + public network access)")
except Exception as e:
    st.error(f"❌OSS configuration error：{e}")
    st.stop()

# ---- 文件上传
files = st.file_uploader("Please select one or more DOC / DOCX format files.", type=["docx"], accept_multiple_files=True)
if not files:
    st.info("Please select DOC / DOCX format files first.")
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
# === NEW: 开始按钮 & 解析进度显示 ===
start = st.button("🚀 开始解析并建立索引", type="primary")
if not start:
    st.stop()

parse_bar = st.progress(0, text="准备解析…")
file_log = st.container()          # 动态滚动显示“当前处理的文件名”
parsed_rows = []                   # 累积显示条目

# ---- 解析 & 抽图（图片会落在本地 MEDIA_DIR）
# ---- 解析 & 抽图（带进度条与文件名滚动日志）
all_clauses: List[Dict[str, Any]] = []
total_files = len(local_paths)
for i, p in enumerate(local_paths, 1):
    cs = parse_docx_into_clauses(p)
    all_clauses.extend(cs)

    parsed_rows.append(f"{i}/{total_files} · {p.name} · 条款 {len(cs)}")
    file_log.write("\n".join(parsed_rows[-30:]))  # 只显示最后30条，避免过长
    parse_bar.progress(i/total_files, text=f"解析进度 {i}/{total_files} · {p.name}")

if not all_clauses:
    st.warning("未解析到条款。")
    st.stop()

# ---- 上传 DOCX 到 A 桶（并生成公网 URL）
st.subheader("上传 DOCX 到 OSS（A 桶）")
docx_url_map: Dict[str, str] = {}
docx_bar = st.progress(0, text="准备上传 DOCX…")

for i, p in enumerate(local_paths, 1):
    key = f"docx/{p.name}"
    oss_put(bucket_docx, p, key)
    u = url_docx(key)
    docx_url_map[p.name] = u
    st.write(f"☁️ {p.name} → {u}")
    docx_bar.progress(i/len(local_paths), text=f"DOCX 上传 {i}/{len(local_paths)} · {p.name}")

# ---- 上传图片到 B 桶，并把 meta 里的 media 改为公网 URL
st.subheader("上传图片到 OSS（B 桶），回填 URL")
uploaded_media: Dict[str, str] = {}

total_imgs = sum(len(c.get("media", [])) for c in all_clauses)
done_imgs = 0
img_bar = st.progress(0, text="准备上传图片…")

for c in all_clauses:
    new_media = []
    for rel in c.get("media", []):
        if rel in uploaded_media:
            new_media.append(uploaded_media[rel])
            done_imgs += 1
            img_bar.progress(done_imgs/max(total_imgs,1), text=f"图片上传 {done_imgs}/{total_imgs}")
            continue

        abs_path = (REPO_ROOT / rel).resolve()
        if not abs_path.exists():
            # 计入进度，避免“卡住”
            done_imgs += 1
            img_bar.progress(done_imgs/max(total_imgs,1), text=f"图片缺失 {done_imgs}/{total_imgs} · {rel}")
            continue

        try:
            rel_key = abs_path.relative_to(MEDIA_DIR).as_posix()
        except Exception:
            rel_key = abs_path.name
        key = f"media/{rel_key}"
        oss_put(bucket_media, abs_path, key)
        u = url_media(key)
        uploaded_media[rel] = u
        new_media.append(u)

        done_imgs += 1
        img_bar.progress(done_imgs/max(total_imgs,1), text=f"图片上传 {done_imgs}/{total_imgs} · {abs_path.name}")

    c["media"] = new_media

# ---- 生成向量 & 构建索引（本地）
st.subheader("Embedding & FAISS")
os.environ["RAG_EMBED_MODEL"] = model_name
model = get_embedder(model_name)
texts = [c["text"] for c in all_clauses]

# === NEW: 手动分批做 embedding，并显示进度 ===
emb_bar = st.progress(0, text="Embedding…")
vec_chunks = []
N = len(texts)
B = int(batch)
for j in range(0, N, B):
    sub = texts[j:j+B]
    arr = model.encode(sub, normalize_embeddings=True)   # 直接调用底层，避免控制台进度条
    vec_chunks.append(arr.astype("float32"))
    emb_bar.progress(min((j+len(sub))/max(N,1), 1.0), text=f"Embedding {j+len(sub)}/{N}")

vecs = np.vstack(vec_chunks)
index = build_faiss_index(vecs, use_gpu_if_possible=not force_cpu)

INDEX_DIR.mkdir(parents=True, exist_ok=True)
faiss_path = INDEX_DIR / "faiss.index"
meta_path  = INDEX_DIR / "meta.jsonl"

st.write("写入索引路径：", faiss_path.resolve())
st.write("写入元数据路径：", meta_path.resolve())

# 写入索引（通过 ingest_docx 内部的 faiss 封装，避免顶层导入 faiss）
write_faiss_index(index, faiss_path)

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
