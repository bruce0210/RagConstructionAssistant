# interface/Home.py
from __future__ import annotations
import re
import os, sys, json
from pathlib import Path
from typing import List, Dict, Any
import streamlit as st
import numpy as np
import textwrap
import html
import time
from streamlit_cookies_manager import EncryptedCookieManager
from streamlit.components.v1 import html as st_html

# ------------------ 轻量路径配置（避免导入重模块） ------------------

st.markdown("""
<style>
/* ma_ui_css_v2 */

/* MA button warm color (orange) */
button[title="Multi-agent Arbitration"] {
  background: linear-gradient(90deg, #ff9f1c, #ffbf69) !important;
  color: #1b1b1b !important;
  border: 1px solid rgba(255,255,255,0.18) !important;
}
button[title="Multi-agent Arbitration"]:hover {
  filter: brightness(1.03);
}

/* Close button smaller to match caption */
button[title="Close MA panel"] {
  padding: 0.18rem 0.55rem !important;
  font-size: 0.86rem !important;
  line-height: 1.1 !important;
}
</style>
""", unsafe_allow_html=True)

REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR  = REPO_ROOT / "data" / "index"

# === Telemetry：从 core 目录引入 ===
CORE_DIR = REPO_ROOT / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.append(str(CORE_DIR))

import importlib
import telemetry as _telemetry
importlib.reload(_telemetry)

from telemetry import (
    ensure_schema as log_ensure,
    start_query, finish_query,
    save_answer, set_reaction,
)

from ma_arbitration import run_ma_for_ui, default_llm_ma_dir
set_hit_reaction = getattr(_telemetry, "set_hit_reaction", lambda *args, **kwargs: None)

log_ensure()  # 建表（存在则跳过）

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

# —— 处理弹窗关闭（新旧 API 兼容）——
try:
    qp = st.query_params
    val = qp.get("close_explain", None)
    if val in ("1", ["1"]):
        st.session_state.show_explanation = False
        qp.clear()
except Exception:
    p = st.experimental_get_query_params()
    if p.get("close_explain") in (["1"], "1"):
        st.session_state.show_explanation = False
        st.experimental_set_query_params()

# —— 答案点赞/点踩 —— 
try:
    qp2 = st.query_params
    react = qp2.get("react", None)
    aid_q = qp2.get("aid", None)
    if react and aid_q:
        aid_val = int(aid_q if isinstance(aid_q, str) else aid_q[0])
        up = react in ("up", ["up"])
        user_id = (st.session_state.get("user") or {}).get("id")
        set_reaction(aid_val, user_id, 1 if up else -1)
        qp2.clear()
        st.toast("已记录：👍 Helpful" if up else "已记录：👎 Unhelpful")
except Exception:
    pass

# —— Top-K 条款“相关/不相关” —— 
try:
    qp3 = st.query_params
    hr = qp3.get("hit_react", None)    # "up" / "down"
    qid_q = qp3.get("qid", None)       # query_id
    clause_q = qp3.get("clause", None) # 条款号
    if hr and qid_q and clause_q:
        qid_val = int(qid_q if isinstance(qid_q, str) else qid_q[0])
        clause_val = clause_q if isinstance(clause_q, str) else clause_q[0]
        user_id = (st.session_state.get("user") or {}).get("id")
        set_hit_reaction(qid_val, clause_val, user_id, 1 if hr in ("up", ["up"]) else -1)
        qp3.clear()
        st.toast("Feedback regarding the relevance of this clause has been recorded~")
except Exception:
    pass

st.set_page_config(
    page_title="RAG Construction Assistant",
    page_icon="🏗️",
    layout="centered"
)

# —— 隐藏 cookies-manager 的同步 iframe（避免侧栏莫名空白）——
st.markdown("""
<style>
iframe[src*="streamlit_cookies_manager.cookie_manager.sync_cookies"]{
  width:0 !important; height:0 !important; min-height:0 !important;
  display:block !important; visibility:hidden !important;
}
</style>
""", unsafe_allow_html=True)

# —— Cookie 管理器 —— 
cookies = EncryptedCookieManager(
    prefix="ragca_",
    password=os.getenv("COOKIES_PASSWORD", "RAGCA_DEMO")
)
cookies_ready = cookies.ready()  # 只读状态，不再 st.stop()

st.title("🏗️🔎 RAG Construction Assistant")
st.caption("Ask questions about building specifications, engineering standards or any construction engineering regulations.")

# —— 若 session 没有 user，则尝试从 cookie 恢复
if "user" not in st.session_state:
    u_raw = cookies.get("user") if cookies_ready else None
    if u_raw:
        try:
            st.session_state["user"] = json.loads(u_raw)
        except Exception:
            pass

with st.sidebar:
    model_name = st.selectbox("Embedding Model",
                              ["BAAI/bge-base-zh-v1.5", "BAAI/bge-m3"], index=1)
    topk = st.slider("Top-K", 1, 10, 5, 1)
    st.markdown("---")

    # —— 侧栏账户区 —— 
    if st.session_state.get("user"):
        u = st.session_state["user"]

        # 标题与头像同块渲染
        st.markdown(
            f'''
            <div style="margin:0 0 10px 0; line-height:1;">
              <div style="font-size:15px;font-weight:700;color:#e5e7eb;">🔐 Account</div>
              <div style="display:flex;align-items:center;gap:10px;margin:14px 0 12px;">
                <img src="https://ragca-project-attachments.oss-ap-northeast-1.aliyuncs.com/default_avatar.png"
                     alt="avatar"
                     style="width:36px;height:36px;border-radius:50%;object-fit:cover;border:3px solid #444;" />
                <div style="line-height:1.2;">
                  <div style="font-size:15px;color:#9ca3af;">Logged in：</div>
                  <div style="font-weight:700;font-size:15px;">{u.get("username","")}</div>
                </div>
              </div>
            </div>
            ''', unsafe_allow_html=True
        )

        if st.button("Log out", key="btn_logout", use_container_width=True):
            # 1) 清 session
            st.session_state.pop("user", None)
            # 2) 立刻清 cookie（在侧栏内就执行，确保下一次 rerun 不会被自动恢复）
            try:
                cookies["user"] = ""
                cookies.save()
            except Exception:
                pass
            # 3) rerun
            st.rerun()

    else:
        st.caption("🔐 Account")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Log in", key="btn_login", use_container_width=True):
                try:
                    st.switch_page("pages/Auth.py")
                except Exception:
                    st.markdown('[前往登录/注册](pages/Auth.py)')
        with col2:
            if st.button("Register", key="btn_register", use_container_width=True):
                try:
                    st.switch_page("pages/Auth.py")
                except Exception:
                    st.markdown('[前往登录/注册](pages/Auth.py)')

    st.markdown("---")
    st.caption("🏗️  Ask questions about building specifications, engineering standards or any construction engineering regulations.")
    st.markdown(
        """
        <a href="https://github.com/bruce0210/rag_construction_assistant" target="_blank">
            <img src="https://github.com/codespaces/badge.svg" alt="Open in GitHub">
        </a>
        """, unsafe_allow_html=True,
    )

# —— 主区：仅当值变化时才写 cookie（稳态；避免重复注入 iframe）——
if cookies_ready:
    want = json.dumps(st.session_state["user"], ensure_ascii=False) if "user" in st.session_state else ""
    curr = cookies.get("user") or ""
    if want != curr:
        try:
            cookies["user"] = want
            cookies.save()
        except Exception:
            pass

faiss_path = INDEX_DIR / "faiss.index"
meta_path  = INDEX_DIR / "meta.jsonl"

# ------------------------- UI 与工具函数 --------------------------
_STATUS_MAP = {
    "现行": ("In Force",  "#16a34a"),
    "废止": ("Obsolete",  "#B22222"),
}
# 英文入参也能兼容
_EN2CN = {
    "in force": "现行",
    "active":   "现行",
    "obsolete": "废止",
    "withdrawn":"废止",
}

def _status_badge(status: str) -> str:
    """
    入参可以是中文(现行/废止)或英文(In Force/Obsolete)。
    前端统一显示英文，并按规范着色。
    """
    cn = status if status in _STATUS_MAP else _EN2CN.get(status.lower(), "")
    if cn in _STATUS_MAP:
        label, color = _STATUS_MAP[cn]
    else:
        label, color = (status or "Unknown"), "#6b7280"

    return (
        f'<span style="background:{color};color:#fff;'
        f'padding:2px 8px;border-radius:6px;font-size:12px;">{label}</span>'
    )


def _doc_title_and_status(src: str) -> tuple[str, str]:
    """
    从文件名中提取标题与状态：
    - 支持中文“现行/废止”，以及带括号的结尾（如：(...)(现行) / （废止））
    - 仍兼容原来的“结尾两个字”写法
    返回的 status 为英文，以便直接用于 UI 显示。
    """
    name = src.split("_", 1)[-1] if "_" in src else src
    no_ext = name[:-5] if name.lower().endswith(".docx") else name

    # 1) 优先匹配结尾括号中的状态：(...)(现行)/(废止)
    m = re.search(r"[（(]?(现行|废止)[)）]?$", no_ext)
    if m:
        status_cn = m.group(1)
        title = re.sub(r"[（(]?(现行|废止)[)）]?$", "", no_ext).rstrip(" -_[]（）()")
    else:
        # 2) 回退到原逻辑：结尾两个字
        tail2 = no_ext[-2:] if len(no_ext) >= 2 else ""
        if tail2 in ("现行", "废止"):
            status_cn = tail2
            title = no_ext[:-2].rstrip(" -_[]（）()")
        else:
            status_cn = ""
            title = no_ext

    status_en = _STATUS_MAP[status_cn][0] if status_cn else ""
    return title, status_en

def render_clause_text(text: str):
    """把条文正文与“条文说明：”分开展示，并给说明加紫色底。"""
    if not text:
        return
    import re
    s = str(text).strip()
    parts = re.split(r'\s*条文说明\s*[:：]\s*', s, maxsplit=1)
    if len(parts) == 2:
        main, note = parts[0], parts[1]
        st.markdown(main)
        st.markdown(
            f'''
            <div style="margin-top:8px; line-height:1.7;">
              <span style="background:#7c3aed; color:#fff; padding:2px 8px; border-radius:6px; font-size:12px;">
                Explanation of the Provisions
              </span>
              <span style="margin-left:.5rem;">{note.strip()}</span>
            </div>
            ''', unsafe_allow_html=True
        )
    else:
        st.markdown(s)

# —— 抽成函数：命中渲染（按钮间距加大）——
def render_hits(hits: list[dict]):
    if not hits:
        return
    for i, r in enumerate(hits, 1):
        with st.container(border=True):
            similarity = r.get("_score", 0.0) * 100
            st.markdown(f"**Top {i}** · Semantic Similarity={similarity:.2f}%")
            render_clause_text(r.get("text"))
            media = r.get("media") or []
            if isinstance(media, list) and media:
                for url in media:
                    st.image(url, use_container_width=True)
            title, status = _doc_title_and_status(r.get("source",""))
            badge = _status_badge(status) if status else ""
            st.markdown("---")
            st.markdown(f"Source Standard: 《{title}》", unsafe_allow_html=True)
            st.markdown(f"Status: {badge}", unsafe_allow_html=True)

            # Top-K 相关性反馈（与底部边框留出空间）
            qid_for_ui = st.session_state.get("last_query_id")
            cl_no = (r.get("clause_no") or "").strip()
            if qid_for_ui and cl_no:
                st.markdown(
                    f'''
                    <div style="margin:12px 0 8px 0; display:flex; gap:10px;">
                      <a href="./?hit_react=up&qid={qid_for_ui}&clause={cl_no}" target="_self"
                         style="text-decoration:none; background:#065f46; color:#fff;
                                padding:2px 6px; border-radius:6px; font-size:12px;">👍 Related</a>
                      <a href="./?hit_react=down&qid={qid_for_ui}&clause={cl_no}" target="_self"
                         style="text-decoration:none; background:#7f1d1d; color:#fff;
                                padding:2px 6px; border-radius:6px; font-size:12px;">👎 Unrelated</a>
                    </div>
                    ''', unsafe_allow_html=True
                )

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
    """严格“有据可查”地生成回答。"""
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

    if level == "Brief Mode":
        length_hint = "目标长度：约150–250字。"; max_tokens = 350
        structure = ("【结论】一句话回答；\n"
                     "【依据】2–4条，逐条标注（条款号｜规范名全称）；\n"
                     "【注意事项】如有则列出；\n")
    elif level == "Advanced Mode":
        length_hint = "目标长度：约600–900字，拒绝空话套话。"; max_tokens = 950
        structure = ("【结论】先给出明确数值/判断；\n"
                     "【条款释义】解释关键术语与阈值含义；\n"
                     "【适用范围/边界与例外】指出适用对象与限制；\n"
                     "【依据】逐条列出并标注；\n"
                     "【计算或校核示例】如适用；\n"
                     "【实施建议/风险提示】2–4 条；\n")
    else:
        length_hint = "目标长度：约300–500字。"; max_tokens = 650
        structure = ("【结论】一句话回答并给出关键数值；\n"
                     "【条款释义】工程表述；\n"
                     "【依据】逐条列出并标注；\n"
                     "【注意事项】列常见边界；\n")

    sys_prompt = (
        "你是建筑工程规范检索助手，严格基于我提供的“命中文本”回答。"
        "不得臆造未出现的数值/条件；若证据不足必须直接说明“依据不足”。\n"
        "写作要求：\n"
        "- 中文输出，术语准确、可执行；\n"
        "- 用编号或短小标题组织；\n"
        "- 每条依据末尾标注（条款号｜规范名全称）；\n"
        f"- {length_hint}\n"
        f"{structure}"
    )

    from openai import OpenAI
    client = OpenAI(api_key=st.secrets["openai"]["api_key"])
    payload = {"query": query, "hits": ctx_blocks}
    resp = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.2,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    return resp.choices[0].message.content.strip()
# -------------------------------------------------------------------

# ---- 从 Prompt Template 页面带来的预填查询（只初始化一次） ----
prefill = st.session_state.pop("home_query_prefill", "")  # 读完即删，防止覆盖

# 首次进入本页时，用预填或默认文案初始化输入框的 session 状态
if "query" not in st.session_state:
    st.session_state["query"] = prefill or "Search: 'How are fire hazards in production classified?' Try it..."

# 绑定到 session 的输入框（不要再传 value）
st.text_input(
    "👷‍♂️How can I help you with your construction project today?",
    key="query"
)

query = st.session_state["query"]

# query = st.text_input("👷‍♂️How can I help you with your construction project today?", "Search: What is BIM? Try it...")
col_go, col_gpt, col_cfg, col_ma = st.columns([0.8, 0.85, 0.3, 0.9])
with col_go:
    auto_go = bool(prefill) and not st.session_state.get("auto_go_ran")
    if auto_go:
        st.session_state["auto_go_ran"] = True
    go = st.button("🚀 Go!", type="primary", use_container_width=True) or auto_go
with col_gpt:
    explain_btn = st.button("🧑‍ Let Prof.LLM Explain", type="secondary", use_container_width=True)
with col_cfg:
    if "detail_level" not in st.session_state:
        st.session_state.detail_level = "Standard Mode"
    if hasattr(st, "popover"):
        with st.popover("⚙️"):
            st.session_state.detail_level = st.radio(
                "Choose the depth of interpretation", ["Brief Mode", "Standard Mode", "Advanced Mode"], index=["Brief Mode","Standard Mode","Advanced Mode"].index(st.session_state.detail_level)
            )
    else:
        with st.expander("⚙️"):
            st.session_state.detail_level = st.radio(
                "Choose the depth of interpretation", ["Brief Mode", "Standard Mode", "Advanced Mode"], index=["Brief Mode","Standard Mode","Advanced Mode"].index(st.session_state.detail_level)
            )
with col_ma:
        ma_btn = st.button("🧩 Multi-agent Arbitration", type="secondary", use_container_width=True)
ma_panel_slot = st.container()  # MA panel fixed under buttons
# —— 固定“解读”spinner 的位置（输入框下方）
explain_spinner_slot = st.empty()
ma_spinner_slot = st.empty()

if "show_explanation" not in st.session_state:
    st.session_state.show_explanation = False
if "explanation_text" not in st.session_state:
    st.session_state.explanation_text = ""
if "last_query_id" not in st.session_state:
    st.session_state.last_query_id = None
if "last_answer_id" not in st.session_state:
    st.session_state.last_answer_id = None
if "ma_show" not in st.session_state:
    st.session_state.ma_show = False
if "ma_result" not in st.session_state:
    st.session_state.ma_result = None
if "ma_error" not in st.session_state:
    st.session_state.ma_error = ""
if "ma_answer_id" not in st.session_state:
    st.session_state.ma_answer_id = None

if go and query.strip():
    user_id = (st.session_state.get("user") or {}).get("id")
    qid = start_query(user_id, query.strip(), topk, model_name, page="home")
    st.session_state.last_query_id = qid
    t0 = time.perf_counter()

    with st.spinner("Searching..."):
        hits = search(query.strip(), topk)

    latency_ms = int((time.perf_counter() - t0) * 1000)
    top_clause_nos = []
    try:
        top_clause_nos = [ (r.get("clause_no") or "").strip() for r in (hits or []) ]
        top_clause_nos = [x for x in top_clause_nos if x]
    except Exception:
        pass
    finish_query(qid, n_hits=len(hits or []), latency_ms=latency_ms,
                 top_clause_nos=top_clause_nos)

    SIM_THRESHOLD = 50.0
    filtered_hits = [r for r in hits if r.get("_score", 0.0) * 100 >= SIM_THRESHOLD]
    st.session_state["last_hits"] = filtered_hits

    if not hits:
        st.info("Oh... No results were found (Please try rephrasing your question~)")
    elif not filtered_hits:
        st.info("The semantic similarity is too low... Please try a different way of asking~")
    else:
        render_hits(filtered_hits)

# —— 非 Go! 的 rerun（比如点了相关/不相关）也保留命中列表 —— 
elif st.session_state.get("last_hits"):
    render_hits(st.session_state["last_hits"])

# —— 解读：spinner 出现在输入框下方（通过占位容器）——
if explain_btn and query.strip():
    with explain_spinner_slot.container():
        prof_llm_slot = st.empty()
        # prof_llm_slot.info("Prof.LLM: Searching...")
        with st.spinner("Searching..."):
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

                    if not st.session_state.get("last_query_id"):
                        user_id = (st.session_state.get("user") or {}).get("id")
                        qid_tmp = start_query(user_id, query.strip(), topk, model_name, page="home")
                        clause_nos_for_llm = []
                        try:
                            clause_nos_for_llm = [ (r.get("clause_no") or "").strip() for r in (hits_for_llm or []) ]
                            clause_nos_for_llm = [x for x in clause_nos_for_llm if x]
                        except Exception:
                            pass
                        finish_query(qid_tmp, n_hits=len(hits_for_llm or []), latency_ms=None,
                                     top_clause_nos=clause_nos_for_llm)
                        st.session_state.last_query_id = qid_tmp

                    ev = {
                        "clause_nos": [ (r.get("clause_no") or "").strip() for r in (hits_for_llm or [])[:5] ],
                        "scores":     [ float(r.get("_score", 0.0)) for r in (hits_for_llm or [])[:5] ],
                    }
                    aid = save_answer(st.session_state.last_query_id, explanation, ev)
                    st.session_state.last_answer_id = aid

                except Exception as e:
                    st.error(f"The engineer is currently making the necessary repairs and will finish soon.：{e}")
            else:
                st.warning("The semantic similarity is too low... Please try a different way of asking~")
        prof_llm_slot.empty()
    # 解读结束后清掉 spinner
    explain_spinner_slot.empty()





# —— 多智能体仲裁（展示 6 专家 + 仲裁；本阶段先不落库）——
if "ma_show" not in st.session_state:
    st.session_state.ma_show = False
if "ma_result" not in st.session_state:
    st.session_state.ma_result = None
if "ma_error" not in st.session_state:
    st.session_state.ma_error = ""

# 固定展示位置（按钮下方）
if "ma_panel_slot" not in globals():
    ma_panel_slot = st.container()
if "ma_spinner_slot" not in globals():
    ma_spinner_slot = st.empty()

# ma_btn 可能在某些分支未定义，这里做保护
try:
    _ma_clicked = bool(ma_btn) and bool(query.strip())
except Exception:
    _ma_clicked = False

# 点击按钮：重新跑一次 MA（使用当前输入框 query）
if _ma_clicked:
    # --- [MA][DB] scheme A: dedicated query_id for this MA run ---
    user_id = (st.session_state.get("user") or {}).get("id")
    ma_model_name = f"{model_name}+MA" if "+MA" not in str(model_name) else str(model_name)
    ma_qid = start_query(user_id, query.strip(), 50, ma_model_name, page="home")
    st.session_state.ma_query_id = ma_qid
    ma_t0 = time.perf_counter()
    st.session_state.ma_show = True
    st.session_state.ma_error = ""
    st.session_state.ma_result = None

    with ma_spinner_slot.container():
        with st.spinner("Running multi-agent arbitration..."):
            try:
                hits_for_ma = search(query.strip(), 50)  # 后端默认 Top 50 candidates
                if not hits_for_ma:
                    raise RuntimeError("No candidates were found. Please try rephrasing your question.")
                llm_ma_dir = default_llm_ma_dir(REPO_ROOT)
# [MA] load env.sh
                # Load API keys into current Streamlit process (strip quotes)
                import os
                from pathlib import Path as _Path
                _env_sh = _Path(llm_ma_dir) / "configs" / "env.sh"
                import os
                os.environ.setdefault("DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
                if _env_sh.exists():
                    for _raw in _env_sh.read_text(encoding="utf-8", errors="ignore").splitlines():
                        _line = _raw.strip()
                        if (not _line) or _line.startswith("#"):
                            continue
                        if _line.startswith("export "):
                            _line = _line[len("export "):].strip()
                        if "=" not in _line:
                            continue
                        _k, _v = _line.split("=", 1)
                        _k = _k.strip()
                        _v = _v.strip().strip('"').strip("'").strip()
                        if _k:
                            os.environ[_k] = _v

                ma_out = run_ma_for_ui(query.strip(), hits_for_ma, llm_ma_dir=llm_ma_dir, candidates_topn=50)
                st.session_state.ma_result = ma_out

                # --- [MA][DB] persist final answer + evidence into PGSQL ---
                try:
                    _final = (ma_out or {}).get("final") or {}
                    _final_answer = _final.get("final_answer") or ""
                    _final_clause_ids = _final.get("final_clause_ids") or []
                    _ev = (ma_out or {}).get("evidence_bundle") or {}
                    if isinstance(_ev, dict):
                        _ev.setdefault("triggered", (ma_out or {}).get("triggered"))
                        _ev.setdefault("used_models", (ma_out or {}).get("used_models"))
                    # normalize to JSON-serializable primitives (telemetry.save_answer uses json.dumps)
                    _ev = json.loads(json.dumps(_ev, ensure_ascii=False, default=str))
                    st.session_state.ma_answer_id = save_answer(ma_qid, _final_answer, _ev)

                    _top_hits = [str(x).strip() for x in (_final_clause_ids or []) if str(x).strip()]
                    if not _top_hits:
                        _top_hits = [str((r.get("clause_no") or r.get("clause_id") or "")).strip() for r in (hits_for_ma or [])]
                        _top_hits = [x for x in _top_hits if x]
                    finish_query(
                        ma_qid,
                        n_hits=len(hits_for_ma or []),
                        latency_ms=int((time.perf_counter() - ma_t0) * 1000),
                        top_clause_nos=_top_hits,
                    )
                except Exception as _db_e:
                    st.session_state.ma_error = (st.session_state.ma_error + f" | DB persist failed: {_db_e}").strip(" |")
            except Exception as e:
                st.session_state.ma_error = str(e)
                try:
                    finish_query(
                        ma_qid,
                        n_hits=0,
                        latency_ms=int((time.perf_counter() - ma_t0) * 1000),
                        top_clause_nos=[],
                    )
                except Exception:
                    pass

    ma_spinner_slot.empty()

# 展示面板：6 专家辩论 + 仲裁
if st.session_state.get("ma_show", False):
    with ma_panel_slot:
        with st.expander("🧩 Multi-agent Arbitration (Debate + Arbitration)", expanded=True):
            col_close, col_note = st.columns([2, 10])
            with col_close:
                if st.button("❌ Close", key="ma_close_panel", help="Close MA panel"):
                    st.session_state.ma_show = False
                    st.session_state.ma_result = None
                    st.session_state.ma_error = ""
            with col_note:
                st.caption("Experts Model: Qwen-max (6 agents) · Arbiter Model: OpenAI (GPT-4o)")

            if st.session_state.get("ma_error"):
                st.error(st.session_state.ma_error)
            elif not st.session_state.get("ma_result"):
                st.info("Click the MA button to run arbitration.")
            else:
                ma_out = st.session_state.ma_result or {}
                diag = ma_out.get("trigger_diag") or {}
                triggered = bool(ma_out.get("triggered", False))

                st.markdown(f"**Arbiter triggered:** `{triggered}`")
                if diag.get("trigger_reasons"):
                    st.markdown("**Trigger reasons:** " + ", ".join(diag.get("trigger_reasons", [])))

                final = ma_out.get("final") or {}
                st.markdown("## ✅ Final Decision")
                if final.get("final_answer"):
                    st.success(final["final_answer"])
                else:
                    st.json(final)
                tabs = st.tabs(["arch", "struct", "plumb", "hvac", "elec", "fire", "final_json"])
                expert_list = ma_out.get("experts") or []
                expert_by = {e.get("agent_id"): e for e in expert_list if isinstance(e, dict) and e.get("agent_id")}
                order = ["arch", "struct", "plumb", "hvac", "elec", "fire"]

                for tab, aid in zip(tabs[:6], order):
                    with tab:
                        obj = expert_by.get(aid)
                        if not obj:
                            st.info("No output.")
                        else:
                            st.json(obj)

                with tabs[6]:
                    st.json(final)

# ---- 弹窗（顶层 DOM 注入，固定在页面中间；无缩进，避免被 Markdown 当作代码块） ----
if st.session_state.get("show_explanation", False):
    if st.button("❌ Close", key="close_explain_top"):
        st.session_state.show_explanation = False
        st.stop()

    # 遮罩（顶层；不拦截点击）—— 注意无任何前导空格
    overlay_html = '<div style="position:fixed;inset:0;background:rgba(0,0,0,0.45);z-index:9998;pointer-events:none;"></div>'
    st.markdown(overlay_html, unsafe_allow_html=True)

    # 只转义 LLM 正文（保留换行）
    ex_html = html.escape(st.session_state.explanation_text).replace("\n", "<br/>")

    # 底部按钮（原生 HTML）
    aid_for_ui = st.session_state.get("last_answer_id")
    btns_html = ""
    if aid_for_ui:
        btns_html = (
            f'<div style="margin-top:12px;display:flex;gap:12px;">'
            f'  <a href="./?react=up&aid={aid_for_ui}" target="_self" '
            f'     style="text-decoration:none;background:#065f46;color:#fff;padding:4px 8px;'
            f'            border-radius:8px;font-size:13px;">👍 Helpful</a>'
            f'  <a href="./?react=down&aid={aid_for_ui}" target="_self" '
            f'     style="text-decoration:none;background:#7f1d1d;color:#fff;padding:4px 8px;'
            f'            border-radius:8px;font-size:13px;">👎 Unhelpful</a>'
            f'</div>'
        )

    # 弹窗主体（无缩进字符串，确保不是代码块）
    modal_html = (
        '<div style="position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);'
        'background-color:#1e1e1e;padding:28px 28px 24px;border-radius:12px;'
        'box-shadow:0 8px 24px rgba(0,0,0,0.7);z-index:9999;width:min(1250px,96vw);'
        'max-height:84vh;overflow-y:auto;pointer-events:auto;">'
        '  <a href="./?close_explain=1" target="_self" title="关闭" '
        '     style="position:absolute;top:8px;right:12px;text-decoration:none;'
        '            background:#374151;color:#fff;padding:2px 8px;border-radius:8px;'
        '            font-weight:700;line-height:1;">×</a>'
        '  <h3 style="color:#00BFFF;margin:0 0 12px 0;">📘 Explanation from the Prof.LLM </h3>'
        f' <div style="color:white;font-size:15px;line-height:1.7;">{ex_html}</div>'
        f' {btns_html}'
        '</div>'
    )
    st.markdown(modal_html, unsafe_allow_html=True)

    if st.button("Close", key="close_explain_bottom"):
        st.session_state.show_explanation = False
        st.experimental_rerun()