# interface/Home.py
from __future__ import annotations
import os, sys, json
from pathlib import Path
from typing import List, Dict, Any
import streamlit as st
import numpy as np

# ------------------ 轻量路径配置（避免导入重模块） ------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR  = REPO_ROOT / "data" / "index"

# 懒加载 + 缓存：模型与索引/元数据，仅在检索时触发
@st.cache_resource(show_spinner=False)
def _load_model_cached(name: str):
    from sentence_transformers import SentenceTransformer
    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", "-1") else "cpu"
    return SentenceTransformer(name, device=device)

@st.cache_data(show_spinner=False)
def _load_meta_cached(p: Path) -> List[Dict, Any]:
    if not p.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows

@st.cache_resource(show_spinner=False)
def _load_faiss_cached(path: Path):
    import faiss
    if not path.exists() or path.stat().st_size == 0:
        return None
    return faiss.read_index(str(path))

def _ensure_index_ready():
    if "records" not in st.session_state:
        st.session_state.records = _load_meta_cached(INDEX_DIR / "meta.jsonl")
    if "index" not in st.session_state:
        st.session_state.index = _load_faiss_cached(INDEX_DIR / "faiss.index")
# -------------------------------------------------------------------

st.set_page_config(
    page_title="RAG Construction Assistant",
    page_icon="🏗️",
    layout="centered"
)

st.title("🏗️🔎 RAG Construction Assistant")
st.caption("Ask questions about building specifications, engineering standards or any construction engineering regulations.")

with st.sidebar:
    model_name = st.selectbox("Embedding Model",
                              ["BAAI/bge-base-zh-v1.5", "BAAI/bge-m3"], index=1)
    topk = st.slider("Top-K", 1, 10, 3, 1)
    st.markdown("---")
    st.caption("🏗️  Ask questions about building specifications, engineering standards or any construction engineering regulations.")
    st.markdown(
        """
        <a href="https://github.com/bruce0210/rag_construction_assistant" target="_blank">
            <img src="https://github.com/codespaces/badge.svg" alt="Open in GitHub">
        </a>
        """,
        unsafe_allow_html=True,
    )

faiss_path = INDEX_DIR / "faiss.index"
meta_path  = INDEX_DIR / "meta.jsonl"

# ------------------------- UI 与工具函数 --------------------------
def _status_badge(status: str) -> str:
    if status == "现行":
        color = "#16a34a"
    elif status == "废止":
        color = "#B22222"
    else:
        color = "#6b7280"
    return f'<span style="background:{color};color:#fff;padding:2px 8px;border-radius:6px;font-size:12px;">{status}</span>'

def _doc_title_and_status(src: str) -> tuple[str, str]:
    name = src.split("_", 1)[-1] if "_" in src else src
    no_ext = name[:-5] if name.lower().endswith(".docx") else name
    status = no_ext[-2:] if len(no_ext) >= 2 else ""
    title = no_ext[:-2] if status in ("现行","废止") else no_ext
    return title, status

def search(q: str, k: int):
    _ensure_index_ready()
    records = st.session_state.get("records", [])
    index = st.session_state.get("index", None)
    if index is None or not records:
        return []

    embedder = _load_model_cached(model_name)
    vec = embedder.encode([q], normalize_embeddings=True).astype("float32")
    import faiss
    D, I = index.search(vec, k)
    out = []
    for rank, idx in enumerate(I[0], 1):
        if 0 <= idx < len(records):
            r = dict(records[idx])
            r["_score"] = float(D[0][rank-1])
            out.append(r)
    return out

# ---------------------- OpenAI（LLM 解读，仅走代理） ----------------
def llm_answer(query: str, hits: list[dict]) -> str:
    """
    用 OpenAI 基于命中的条款生成严格“有据可查”的回答。
    只允许引用 hits 中提供的文本；不允许自创内容。
    """
    ctx_blocks = []
    for r in hits[:3]:
        title, status = _doc_title_and_status(r.get("source",""))
        media = r.get("media") or []
        ctx_blocks.append({
            "clause_no": r.get("clause_no",""),
            "title": title,
            "status": status,
            "text": (r.get("text") or "").strip(),
            "media": media[:5],
        })

    sys_prompt = (
        "你是建筑工程规范检索助手。必须严格基于提供的“命中文本”回答，"
        "不要编造规范条款。输出结构：\n"
        "【结论】一句话回答；\n"
        "【依据】逐条列出并标注条款号；\n"
        "【注意事项】如有则列出。\n"
    )

    import httpx
    from openai import OpenAI

    def _make_http_client(proxy_url: str | None) -> httpx.Client:
        """
        兼容 httpx 0.27 (proxies=) 与 0.28+ (proxy=) 的写法。
        只在 LLM 调用时使用代理。
        """
        timeout = httpx.Timeout(30.0)
        if not proxy_url:
            return httpx.Client(timeout=timeout)

        # 优先尝试旧参名（0.27 及以下）
        try:
            return httpx.Client(proxies=proxy_url, timeout=timeout)
        except TypeError:
            # 回退到新参名（0.28+）
            return httpx.Client(proxy=proxy_url, timeout=timeout)

    # 代理配置（只在 LLM 调用时使用）
    proxy = st.secrets.get("openai", {}).get("proxy")  # 例如 "http://user:pass@127.0.0.1:7890"
    http_client = _make_http_client(proxy)

    client = OpenAI(
        api_key=st.secrets["openai"]["api_key"],
        http_client=http_client
    )

    payload = {"query": query, "hits": ctx_blocks}
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    return resp.choices[0].message.content.strip()
# -------------------------------------------------------------------

query = st.text_input("👷‍♂️How can I help you with your construction project today?", "导线的设计安全系数不应小于多少?")
col_go, col_gpt = st.columns([1,1])
with col_go:
    go = st.button("🚀 Go!", type="primary", use_container_width=True)
with col_gpt:
    explain_btn = st.button("🤖 LLM 解读", type="secondary", use_container_width=True)

if "show_explanation" not in st.session_state:
    st.session_state.show_explanation = False
if "explanation_text" not in st.session_state:
    st.session_state.explanation_text = ""

if go and query.strip():
    with st.spinner("Searching..."):
        hits = search(query.strip(), topk)
        st.session_state["last_hits"] = hits
    if not hits:
        st.info("啊哦...没有检索到结果（请尝试换个问法~）")
    else:
        for i, r in enumerate(hits, 1):
            with st.container(border=True):
                similarity = r.get("_score", 0.0) * 100
                st.markdown(f"**Top {i}** · 语义检索相似度={similarity:.2f}%")
                st.write((r.get("text") or "").strip())
                media = r.get("media") or []
                if isinstance(media, list) and media:
                    for url in media:
                        st.image(url, use_container_width=True)
                title, status = _doc_title_and_status(r.get("source",""))
                badge = _status_badge(status) if status else ""
                st.markdown("---")
                st.markdown(f"本条款出自规范：《{title}》", unsafe_allow_html=True)
                st.markdown(f"该规范当前实施状态：{badge}", unsafe_allow_html=True)

if explain_btn and query.strip():
    with st.spinner("LLM 正在生成结构化解读…"):
        hits_for_llm = st.session_state.get("last_hits")
        if not hits_for_llm:
            hits_for_llm = search(query.strip(), topk)
            st.session_state["last_hits"] = hits_for_llm
        if hits_for_llm:
            try:
                explanation = llm_answer(query.strip(), hits_for_llm)
                st.session_state.explanation_text = explanation
                st.session_state.show_explanation = True
            except Exception as e:
                st.error(f"调用 GPT 失败：{e}")
        else:
            st.warning("没有检索到条款，无法生成解读。")

if st.session_state.get("show_explanation", False):
    st.markdown(
        """
        <div style="
            position: fixed; inset: 0;
            background: rgba(0,0,0,0.45);
            z-index: 9998;
        "></div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div style="
            position: fixed; top: 50%; left: 50%;
            transform: translate(-50%, -50%);
            background-color: #1e1e1e;
            padding: 28px;
            border-radius: 12px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.7);
            z-index: 9999;
            width: min(900px, 88vw);
            max-height: 80vh; overflow-y: auto;
        ">
            <h3 style="color:#00BFFF; margin: 0 0 12px 0;">📘 LLM 解读</h3>
            <div style="color:white; font-size:15px; line-height:1.7;">
        """
        + st.session_state.explanation_text.replace("\n","<br/>")
        + """
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    if st.button("❌ 关闭解读窗口"):
        st.session_state.show_explanation = False
