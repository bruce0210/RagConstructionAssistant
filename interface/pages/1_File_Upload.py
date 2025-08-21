# interface/pages/1_File_Upload.py
from __future__ import annotations
import os, sys, time, json
from pathlib import Path
from typing import List, Dict, Any
from shutil import rmtree

import streamlit as st
import numpy as np
import pandas as pd  # 用于官方 dataframe 预览

# ---- small utils for cleanup ----
def _safe_unlink(p: Path):
    try:
        p.unlink(missing_ok=True)
    except Exception:
        pass

def _safe_rmtree(p: Path):
    try:
        rmtree(p, ignore_errors=True)
    except Exception:
        pass

def _fmt_bytes(n: int | None) -> str:
    if not isinstance(n, (int, float)) or n < 0:
        return "-"
    if n < 1024:
        return f"{n:.0f} B"
    if n < 1024**2:
        return f"{n/1024:.1f} KB"
    if n < 1024**3:
        return f"{n/1024**2:.1f} MB"
    return f"{n/1024**3:.1f} GB"
# ----------------------------------

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
ingest_docx = importlib.reload(ingest_docx)  # 强制使用最新版本
# 从模块对象上取属性
REPO_ROOT  = ingest_docx.REPO_ROOT
INDEX_DIR  = ingest_docx.INDEX_DIR
MEDIA_DIR  = ingest_docx.MEDIA_DIR
parse_docx_into_clauses = ingest_docx.parse_docx_into_clauses
get_embedder            = ingest_docx.get_embedder
embed_texts             = ingest_docx.embed_texts
build_faiss_index       = ingest_docx.build_faiss_index
write_faiss_index       = ingest_docx.write_faiss_index
# 追加模式 + 整文档去重
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
        index=1  # 默认 m3
    )
    batch = st.number_input("Embed batch size", 8, 1024, 512, 8)
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

# ---- 隐藏 uploader 自带的文件卡片
st.markdown("""
<style>
[data-testid="stFileUploader"] [data-testid="stFileUploaderFileList"] { display:none; }
[data-testid="stFileUploader"] .uploadedFile { display:none; }
</style>
""", unsafe_allow_html=True)

# ---- 文件上传（多选）
files = st.file_uploader(
    "Please select one or more DOC / DOCX format files.",
    type=["docx"],
    accept_multiple_files=True
)
if not files:
    st.info("Please select DOC / DOCX format files first.")
    st.stop()

# 官方 DataFrame 预览
rows = [{"File": Path(f.name).name, "Size": _fmt_bytes(getattr(f, "size", None))} for f in files]
df = pd.DataFrame(rows)
df.index = pd.RangeIndex(start=1, stop=len(df)+1, step=1, name="#")
st.dataframe(df, use_container_width=True, height=360)

start_ingest = st.button("🚀 开始处理并建立索引", type="primary")
if not start_ingest:
    st.stop()

# ---- 临时落地（仅点击后写磁盘）
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

# ---- 目标索引/元数据路径
INDEX_DIR.mkdir(parents=True, exist_ok=True)
faiss_path = INDEX_DIR / "faiss.index"
meta_path  = INDEX_DIR / "meta.jsonl"

# === 解析 & 抽图（追加模式 + 整文档去重） ===
parse_bar = st.progress(0, text="准备解析…")
file_log = st.empty()
status_lines: List[str] = []

all_clauses: List[Dict[str, Any]] = []
kept_paths: List[Path] = []
seen_docs = load_seen_doc_hashes(meta_path)
batch_seen_docs: set[str] = set()

total_files = len(local_paths)
for i, p in enumerate(local_paths, 1):
    # 整文档指纹
    try:
        doc_bytes = p.read_bytes()
    except Exception:
        doc_bytes = b""
    doc_hash = file_blake2b_hex(doc_bytes)

    # 已入库 / 本批重复 → 跳过
    if doc_hash in seen_docs or doc_hash in batch_seen_docs:
        status_lines.append(f"{i}/{total_files} · {p.name} · 已跳过（重复文档）")
        file_log.text("\n".join(status_lines[-30:]))
        parse_bar.progress(i/total_files, text=f"解析进度 {i}/{total_files} · {p.name} · 跳过重复")
        continue

    cs = parse_docx_into_clauses(p)
    for c in cs:
        c["doc_hash"] = doc_hash  # 写入到每条记录
    all_clauses.extend(cs)
    kept_paths.append(p)
    batch_seen_docs.add(doc_hash)

    status_lines.append(f"{i}/{total_files} · {p.name} · 条款 {len(cs)}")
    file_log.text("\n".join(status_lines[-30:]))
    parse_bar.progress(i/total_files, text=f"解析进度 {i}/{total_files} · {p.name}")

if not all_clauses:
    st.info("本批文档均为重复或未解析到有效条款，未进行追加。")
    st.stop()

# ---- 上传 DOCX 到 A 桶（仅新文档）
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

# ---- 上传图片到 B 桶，并把 meta 里的 media 改为公网 URL（仅新条目）
st.subheader("上传图片到 OSS（B 桶），回填 URL")
uploaded_media: Dict[str, str] = {}
doc_media_roots: set[Path] = set()  # 本批每个 docx 的 data/media/<slug>

total_imgs = sum(len(c.get("media", [])) for c in all_clauses)
done_imgs = 0
img_bar = st.progress(0, text="准备上传图片…")

for c in all_clauses:
    new_media = []
    for rel in c.get("media", []):
        # 1) 同一批内重复使用同一张 → 直接复用 URL
        if rel in uploaded_media:
            new_media.append(uploaded_media[rel])
            done_imgs += 1
            img_bar.progress(done_imgs/max(total_imgs,1), text=f"图片上传 {done_imgs}/{total_imgs}")
            continue

        # 2) 计算绝对路径
        abs_path = (REPO_ROOT / rel).resolve()
        if not abs_path.exists():
            done_imgs += 1
            img_bar.progress(done_imgs/max(total_imgs,1), text=f"图片缺失 {done_imgs}/{total_imgs} · {rel}")
            continue

        # 记录该文档的图片根目录：data/media/<slug>
        try:
            rel_from_media = abs_path.relative_to(MEDIA_DIR)   # "<slug>/clause_x/img_0.jpg"
            doc_root = MEDIA_DIR / rel_from_media.parts[0]      # data/media/<slug>
            doc_media_roots.add(doc_root)
        except Exception:
            pass

        # 3) 上传并生成公网 URL
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

    # ★ 用 OSS URL 覆盖原本的相对路径
    c["media"] = new_media

# ---- 生成向量（仅新条目） & 追加索引（本地）
st.subheader("Embedding & FAISS (Append Mode)")
os.environ["RAG_EMBED_MODEL"] = model_name
model = get_embedder(model_name)
texts = [c["text"] for c in all_clauses]

emb_bar = st.progress(0, text="Embedding…")
vec_chunks = []
N = len(texts)
B = int(batch)
for j in range(0, N, B):
    sub = texts[j:j+B]
    arr = model.encode(sub, normalize_embeddings=True)
    vec_chunks.append(arr.astype("float32"))
    emb_bar.progress(min((j+len(sub))/max(N,1), 1.0), text=f"Embedding {j+len(sub)}/{N}")
vecs = np.vstack(vec_chunks)

# 旧库存在则在其后追加；否则新建
index = build_or_append_faiss_index(
    vecs, faiss_path, use_gpu_if_possible=not force_cpu
)
write_faiss_index(index, faiss_path)

# ---- 生成/追加元数据（补充 source_url）
for c in all_clauses:
    src = c.get("source", "")
    hit = None
    for p in kept_paths:
        if src == p.name or src == p.name.split("_", 1)[-1]:
            hit = docx_url_map.get(p.name)
            break
    if hit:
        c["source_url"] = hit

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

# ---- 统一清理：临时 DOCX 与本批图片目录（已将 URL 回填为 OSS）
for p in local_paths:
    _safe_unlink(p)
try:
    TMP.rmdir()
except OSError:
    pass

for d in doc_media_roots:
    _safe_rmtree(d)

st.balloons()
st.success("🎉 完成：DOCX 入 A 桶、图片入 B 桶（media 字段为公网 URL）、索引已追加（C 桶备份可选）。")
