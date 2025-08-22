# interface/Home.py
from __future__ import annotations
import os, sys, json
from pathlib import Path
from typing import List, Dict, Any
import streamlit as st
import numpy as np
import textwrap

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
def _load_meta_cached(p: Path, mtime: float, size: int) -> List[Dict[str, Any]]:
    if not p.exists() or size == 0:
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
def _load_faiss_cached(path: Path, mtime: float, size: int):
    import faiss
    if not path.exists() or size == 0:
        return None
    return faiss.read_index(str(path))

def _ensure_index_ready():
    meta_file = INDEX_DIR / "meta.jsonl"
    idx_file  = INDEX_DIR / "faiss.index"
    meta_stat = meta_file.stat() if meta_file.exists() else None
    idx_stat  = idx_file.stat()  if idx_file.exists()  else None

    st.session_state["records"] = _load_meta_cached(
        meta_file,
        meta_stat.st_mtime if meta_stat else 0.0,
        meta_stat.st_size  if meta_stat else 0,
    )
    st.session_state["index"] = _load_faiss_cached(
        idx_file,
        idx_stat.st_mtime if idx_stat else 0.0,
        idx_stat.st_size  if idx_stat else 0,
    )

# -------------------------------------------------------------------

# —— 处理弹窗关闭的 query 参数（新旧 API 兼容）——
try:
    qp = st.query_params           # 新版
    val = qp.get("close_explain", None)
    if val in ("1", ["1"]):
        st.session_state.show_explanation = False
        qp.clear()
except Exception:
    p = st.experimental_get_query_params()
    if p.get("close_explain") in (["1"], "1"):
        st.session_state.show_explanation = False
        st.experimental_set_query_params()


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
    topk = st.slider("Top-K", 1, 10, 5, 1)
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

def render_clause_text(text: str):
    """
    把条文正文与“条文说明：”分开展示，并给说明加紫底。
    兼容：全/半角冒号、前后空格、换行等。
    """
    if not text:
        return
    import re
    s = str(text).strip()

    # 以首个“条文说明：/条文说明:”为界拆分
    parts = re.split(r'\s*条文说明\s*[:：]\s*', s, maxsplit=1)
    if len(parts) == 2:
        main, note = parts[0], parts[1]
        st.markdown(main)  # 正文
        st.markdown(
            f'''
            <div style="margin-top:8px; line-height:1.7;">
              <span style="background:#7c3aed; color:#fff; padding:2px 8px; border-radius:6px; font-size:12px;">
                条文说明
              </span>
              <span style="margin-left:.5rem;">{note.strip()}</span>
            </div>
            ''',
            unsafe_allow_html=True
        )
    else:
        st.markdown(s)

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

# ---------------------- OpenAI（LLM 解读） ----------------
def llm_answer(query: str, hits: list[dict], level: str = "标准") -> str:
    """
    用 OpenAI 基于命中的条款生成严格“有据可查”的回答。
    只允许引用 hits 中提供的文本；不允许自创内容。
    根据 level 控制输出详略。
    """
    # 上下文：最多取前5条命中，避免太短
    use_n = min(5, len(hits))
    ctx_blocks = []
    for r in hits[:use_n]:
        title, status = _doc_title_and_status(r.get("source",""))
        media = r.get("media") or []
        ctx_blocks.append({
            "clause_no": r.get("clause_no",""),
            "title": title,
            "status": status,
            "text": (r.get("text") or "").strip(),
            "media": media[:5],
        })

    # 不同详略的“目标长度 + 结构”
    if level == "敷衍版":
        length_hint = "目标长度：约150–250字。"
        structure = (
            "【结论】一句话回答；\n"
            "【依据】2–4条，逐条标注条款号（格式：条款号｜规范名全称）；\n"
            "【注意事项】如有则列出；\n"
        )
        max_tokens = 350
    elif level == "冒烟版":
        length_hint = "目标长度：约600–900字，拒绝空话套话。"
        structure = (
            "【结论】先给出明确数值/判断；\n"
            "【条款释义】解释关键术语与阈值含义；\n"
            "【适用范围/边界与例外】指出适用对象、工况限制、与何者不适用；\n"
            "【依据】逐条列出，末尾用（条款号｜规范名全称）标注；\n"
            "【计算或校核示例】如该问题涉及验算，给出步骤与判据（无则说明不适用）；\n"
            "【实施建议/风险提示】从设计/施工/运维角度给2–4条可执行建议；\n"
        )
        max_tokens = 950
    else:  # 标准版
        length_hint = "目标长度：约300–500字。"
        structure = (
            "【结论】一句话回答并给出关键数值；\n"
            "【条款释义】用工程表述解释条文要点；\n"
            "【依据】逐条列出并标注（条款号｜规范名全称）；\n"
            "【注意事项】列出常见误区/边界；\n"
        )
        max_tokens = 650

    sys_prompt = (
        "你是建筑工程规范检索助手，严格基于我提供的“命中文本”回答。"
        "不得臆造未出现的数值/条件；若证据不足必须直接说明“依据不足”。\n"
        "写作要求：\n"
        "- 中文输出，面向工程师，术语准确、可执行；\n"
        "- 用编号或短小标题组织，避免空话；\n"
        "- 引用规范时在每条末尾标注（条款号｜规范名全称）；\n"
        f"- {length_hint}\n"
        "输出结构如下（没有的部分简要说明原因，不要硬编）：\n"
        f"{structure}"
    )

    from openai import OpenAI
    client = OpenAI(api_key=st.secrets["openai"]["api_key"])

    payload = {"query": query, "hits": ctx_blocks}
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    return resp.choices[0].message.content.strip()
# -------------------------------------------------------------------

query = st.text_input("👷‍♂️How can I help you with your construction project today?", "搜：建筑物等电位联结导体的最小截面积？ 试试看...")
# 放在 query 输入框下面、Go/解读按钮处
col_go, col_gpt, col_cfg = st.columns([1, 1, 0.2])
with col_go:
    go = st.button("🚀 Go!", type="primary", use_container_width=True)
with col_gpt:
    explain_btn = st.button("🧑‍ 让尹老师解读", type="secondary", use_container_width=True)
with col_cfg:
    # 默认值（只设置一次）
    if "detail_level" not in st.session_state:
        st.session_state.detail_level = "标准版"
    # 用 Streamlit 自带 popover；旧版本没有则退化为 expander
    if hasattr(st, "popover"):
        with st.popover("⚙️"):
            st.session_state.detail_level = st.radio(
                "选择解读深度", ["敷衍版", "标准版", "冒烟版"], index=["敷衍版","标准版","冒烟版"].index(st.session_state.detail_level)
            )
    else:
        with st.expander("⚙️"):
            st.session_state.detail_level = st.radio(
                "选择解读深度", ["敷衍版", "标准版", "冒烟版"], index=["敷衍版","标准版","冒烟版"].index(st.session_state.detail_level)
            )

if "show_explanation" not in st.session_state:
    st.session_state.show_explanation = False
if "explanation_text" not in st.session_state:
    st.session_state.explanation_text = ""

if go and query.strip():
    with st.spinner("Searching..."):
        hits = search(query.strip(), topk)

    SIM_THRESHOLD = 50.0  # 语义相似度阈值（百分比）
    filtered_hits = [r for r in hits if r.get("_score", 0.0) * 100 >= SIM_THRESHOLD]
    st.session_state["last_hits"] = filtered_hits  # 供 LLM 解读使用

    if not hits:
        st.info("啊哦...没有检索到结果（请尝试换个问法~）")
    elif not filtered_hits:
        st.info("语义相似度太低，请换个问法")
    else:
        for i, r in enumerate(filtered_hits, 1):
            with st.container(border=True):
                similarity = r.get("_score", 0.0) * 100
                st.markdown(f"**Top {i}** · 语义检索相似度={similarity:.2f}%")
                render_clause_text(r.get("text"))
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
    with st.spinner("尹老师正在拼命解读…"):
        hits_for_llm = st.session_state.get("last_hits")
        if not hits_for_llm:
            raw_hits = search(query.strip(), topk)
            SIM_THRESHOLD = 60.0
            hits_for_llm = [r for r in raw_hits if r.get("_score", 0.0) * 100 >= SIM_THRESHOLD]
            st.session_state["last_hits"] = hits_for_llm

        if hits_for_llm:
            try:
                explanation = llm_answer(query.strip(), hits_for_llm, st.session_state.detail_level)
                st.session_state.explanation_text = explanation
                st.session_state.show_explanation = True
            except Exception as e:
                st.error(f"召唤尹老师失败，已帮你Call他了，一会就回来~：{e}")
        else:
            st.warning("语义相似度太低，请换个问法")

# ---- 改良版弹窗（遮罩不拦截点击，提供顶部+底部关闭按钮） ----
if st.session_state.get("show_explanation", False):
    # 顶部关闭（先渲染，点击后立即停止）
    if st.button("❌ 关闭解读窗口", key="close_explain_top"):
        st.session_state.show_explanation = False
        st.stop()

    # 背景遮罩仅做视觉，不拦截点击
    # 背景遮罩（确认这段存在 pointer-events: none）
    st.markdown(
        """
        <div style="
            position: fixed; inset: 0;
            background: rgba(0,0,0,0.45);
            z-index: 9998;
            pointer-events: none;
        "></div>
        """,
        unsafe_allow_html=True
    )

    # 中心弹窗（一定要有 unsafe_allow_html=True）
    modal_html = textwrap.dedent("""
    <div style="
      position: fixed; top: 50%; left: 50%;
      transform: translate(-50%, -50%);
      background-color: #1e1e1e;
      padding: 28px 28px 24px 28px;
      border-radius: 12px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.7);
      z-index: 9999;
      width: min(1250px, 96vw);   /* 调这里即可增大宽度 */
      max-height: 84vh;           /* 稍微加高一些可视区域 */
      overflow-y: auto;
    ">
      <!-- 右上角关闭：当前页关闭，不新开窗口 -->
      <a href="./?close_explain=1" target="_self" title="关闭"
         style="position:absolute; top:8px; right:12px; text-decoration:none;
                background:#374151; color:#fff; padding:2px 8px; border-radius:8px;
                font-weight:700; line-height:1;">×</a>

      <h3 style="color:#00BFFF; margin: 0 0 12px 0;">📘 来自尹老师的解读</h3>
      <div style="color:white; font-size:15px; line-height:1.7;">
    """) + st.session_state.explanation_text.replace("\n", "<br/>") + textwrap.dedent("""
      </div>
    </div>
    """)

    st.markdown(modal_html, unsafe_allow_html=True)


    # 底部关闭按钮（可选）
    if st.button("关闭", key="close_explain_bottom"):
        st.session_state.show_explanation = False
        st.experimental_rerun()
