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
build_faiss_index       = ingest_docx.build_faiss_index  # 仍保留，如需新建
write_faiss_index       = ingest_docx.write_faiss_index
# ✅ 追加模式 + 整文档去重：新增工具函数引用
load_seen_doc_hashes        = ingest_docx.load_seen_doc_hashes
file_blake2b_hex            = ingest_docx.file_blake2b_hex
count_existing_meta_lines   = ingest_docx.count_existing_meta_lines
build_or_append_faiss_index = ingest_docx.build_or_append_faiss_index
write_meta_jsonl            = ingest_docx.write_meta_jsonl
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
        index=1  # 默认选 m3，如需保持原样可改回 0
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

# ---- 目标索引/元数据路径（供去重与追加使用）
INDEX_DIR.mkdir(parents=True, exist_ok=True)
faiss_path = INDEX_DIR / "faiss.index"
meta_path  = INDEX_DIR / "meta.jsonl"

# === NEW: 解析 & 抽图（追加模式 + 整文档去重） ===
parse_bar = st.progress(0, text="准备解析…")
file_log = st.container()          # 动态滚动显示“当前处理的文件名”
parsed_rows = []                   # 累积显示条目

all_clauses: List[Dict[str, Any]] = []
kept_paths: List[Path] = []
seen_docs = load_seen_doc_hashes(meta_path)
batch_seen_docs: set[str] = set()

total_files = len(local_paths)
for i, p in enumerate(local_paths, 1):
    # 计算整文档指纹，用于整文档去重
    try:
        doc_bytes = p.read_bytes()
    except Exception:
        doc_bytes = b""
    doc_hash = file_blake2b_hex(doc_bytes)

    # 重复判定：已入库 or 本批已见 → 跳过
    if doc_hash in seen_docs or doc_hash in batch_seen_docs:
        parsed_rows.append(f"{i}/{total_files} · {p.name} · 已跳过（重复文档）")
        file_log.write("\n".join(parsed_rows[-30:]))
        parse_bar.progress(i/total_files, text=f"解析进度 {i}/{total_files} · {p.name} · 跳过重复")
        continue

    cs = parse_docx_into_clauses(p)
    # 给本文件的所有条目打上 doc_hash（便于后续再去重）
    for c in cs:
        c["doc_hash"] = doc_hash
    all_clauses.extend(cs)
    kept_paths.append(p)
    batch_seen_docs.add(doc_hash)

    parsed_rows.append(f"{i}/{total_files} · {p.name} · 条款 {len(cs)}")
    file_log.write("\n".join(parsed_rows[-30:]))  # 只显示最后30条，避免过长
    parse_bar.progress(i/total_files, text=f"解析进度 {i}/{total_files} · {p.name}")

if not all_clauses:
    st.info("本批文档均为重复或未解析到有效条款，未进行追加。")
    st.stop()

# ---- 上传 DOCX 到 A 桶（仅上传非重复的 kept_paths）
st.subheader("上传 DOCX 到 OSS（A 桶）")
docx_url_map: Dict[str, str] = {}
docx_bar = st.progress(0, text="准备上传 DOCX…")

if not kept_paths:
    st.info("本批无新文档需要上传到 A 桶。")
else:
    for i, p in enumerate(kept_paths, 1):
        key = f"docx/{p.name}"
        oss_put(bucket_docx, p, key)
        u = url_docx(key)
        docx_url_map[p.name] = u
        st.write(f"☁️ {p.name} → {u}")
        docx_bar.progress(i/len(kept_paths), text=f"DOCX 上传 {i}/{len(kept_paths)} · {p.name}")

# ---- 上传图片到 B 桶，并把 meta 里的 media 改为公网 URL（仅对新条目）
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

# ---- 生成向量（仅新条目） & 以“追加模式”构建索引（本地）
st.subheader("Embedding & FAISS (Append Mode)")
os.environ["RAG_EMBED_MODEL"] = model_name
model = get_embedder(model_name)
texts = [c["text"] for c in all_clauses]

# 手动分批做 embedding，并显示进度
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

# ✅ 读取旧索引并在其后追加；若没有旧库则新建
index = build_or_append_faiss_index(
    vecs, faiss_path, use_gpu_if_possible=not force_cpu
)
write_faiss_index(index, faiss_path)

# ---- 生成/追加元数据（补充 source_url、doc_hash 已在解析时打上）
# 先把 source_url 回填
for c in all_clauses:
    src = c.get("source", "")
    hit = None
    for p in kept_paths:
        if src == p.name or src == p.name.split("_", 1)[-1]:
            hit = docx_url_map.get(p.name)
            break
    if hit:
        c["source_url"] = hit

# 以追加模式写入 meta.jsonl，并按已有行数续号 id
base_id = count_existing_meta_lines(meta_path)
write_meta_jsonl(all_clauses, meta_path, base_id=base_id, append=True)

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
st.success("🎉 完成：DOCX 入 A 桶、图片入 B 桶、索引已追加（C 桶备份可选）。")
