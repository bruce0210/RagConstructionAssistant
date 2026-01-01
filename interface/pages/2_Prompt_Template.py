# interface/pages/2_Prompt_Template.py
import os, json, re, textwrap, time
from datetime import datetime
import streamlit as st
import psycopg2, psycopg2.extras

# ---------------- 页面配置 ----------------
st.set_page_config(page_title="Prompt Template", page_icon="🧭", layout="centered")
st.title("🧭 Prompt Template")
st.caption("Build and test customized prompt templates for construction queries.")

with st.sidebar:
    st.caption("🧭 Explore and customize prompt templates.")
    st.markdown(
        """
        <a href="https://github.com/bruce0210/rag_construction_assistant" target="_blank">
            <img src="https://github.com/codespaces/badge.svg" alt="Open in GitHub">
        </a>
        """, unsafe_allow_html=True,
    )

# ---------------- 数据库（含自动迁移） ----------------
def get_conn():
    dsn = None
    try:
        if "postgres" in st.secrets and "dsn" in st.secrets["postgres"]:
            dsn = st.secrets["postgres"]["dsn"]
        elif "DATABASE_URL" in st.secrets:
            dsn = st.secrets["DATABASE_URL"]
    except Exception:
        pass
    if not dsn:
        dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("未配置数据库连接。请设置 [postgres].dsn 或环境变量 DATABASE_URL")
    return psycopg2.connect(dsn)

def ensure_schema():
    ddl = """
    CREATE SCHEMA IF NOT EXISTS app;

    CREATE TABLE IF NOT EXISTS app.prompt_runs (
        id               BIGSERIAL PRIMARY KEY,
        user_id          UUID,
        username         TEXT,
        template_key     TEXT NOT NULL,
        template_title   TEXT NOT NULL,
        variables        JSONB,
        input_text       TEXT,
        output_prompt    TEXT,
        polished_by_llm  BOOLEAN NOT NULL DEFAULT FALSE,
        lang             TEXT,
        created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
        page             TEXT
    );

    -- 自动迁移（已有表也补齐缺列）
    ALTER TABLE app.prompt_runs ADD COLUMN IF NOT EXISTS polished_by_llm BOOLEAN NOT NULL DEFAULT FALSE;
    ALTER TABLE app.prompt_runs ADD COLUMN IF NOT EXISTS lang TEXT;
    ALTER TABLE app.prompt_runs ADD COLUMN IF NOT EXISTS page TEXT;

    -- 索引
    CREATE INDEX IF NOT EXISTS idx_prompt_runs_user    ON app.prompt_runs(user_id);
    CREATE INDEX IF NOT EXISTS idx_prompt_runs_created ON app.prompt_runs(created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_prompt_runs_tpl     ON app.prompt_runs(template_key);
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(ddl); conn.commit()

def insert_prompt_run(row: dict):
    sql = """
    INSERT INTO app.prompt_runs
      (user_id, username, template_key, template_title, variables, input_text,
       output_prompt, polished_by_llm, lang, page)
    VALUES (%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s,%s)
    RETURNING id, created_at
    """
    with get_conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, (
            row.get("user_id"), row.get("username"),
            row.get("template_key"), row.get("template_title"),
            json.dumps(row.get("variables") or {}, ensure_ascii=False),
            row.get("input_text"), row.get("output_prompt"),
            bool(row.get("polished_by_llm", False)),
            row.get("lang") or "ZH",
            row.get("page") or "prompt_template",
        ))
        rec = cur.fetchone(); conn.commit()
        return dict(rec)

def list_recent(limit=20, kw: str | None = None, only_me: bool = False, me_id=None):
    where, args = [], []
    if kw:
        where.append("(template_title ILIKE %s OR template_key ILIKE %s OR output_prompt ILIKE %s)")
        args += [f"%{kw}%", f"%{kw}%", f"%{kw}%"]
    if only_me and me_id:
        where.append("user_id = %s"); args.append(me_id)
    sql = """
    SELECT id, created_at, username, template_title, template_key,
           LEFT(output_prompt, 180) AS preview
    FROM app.prompt_runs
    """
    if where: sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY created_at DESC LIMIT %s"; args.append(limit)
    with get_conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, args); return cur.fetchall()

ensure_schema()

# ---------------- 简单模式算法（把自然语言 → 更准的检索短句） ----------------
STOPWORDS_ZH = set("的 了 和 与 及 或 在 对 于 为 是 吗 呢 啊 把 被 就 还 很 比 较 是否 怎么 怎么样 如何 请 问 最 需要".split())
STOPWORDS_EN = set("the a an of for to is are be in on at by with about what how why whether need minimum".split())

DISCIPLINE_HINTS = {
    "General / 综合": [],
    "Electrical / 电气": ["照明", "配电", "接地", "应急", "电缆", "变配电"],
    "Fire / 消防": ["火灾", "疏散", "耐火", "喷淋", "报警", "防火分区"],
    "HVAC / 暖通": ["通风", "空调", "风量", "新风", "排烟", "热回收"],
    "Plumbing / 给排水": ["给水", "排水", "虹吸", "喷淋", "水压", "水泵"],
}

REGION_CODES = {
    "CN (GB/JGJ/CECS)": ["GB", "JGJ", "CECS", "GB/T"],
    "EU (EN/IEC)": ["EN", "IEC"],
    "US (NFPA/ASHRAE/IPC)": ["NFPA", "ASHRAE", "IPC"],
}

def normalize_text(s: str):
    s = re.sub(r"[^\w\u4e00-\u9fa5\-\/\.#]+", " ", s, flags=re.I)
    return re.sub(r"\s+", " ", s).strip()

def keywords_from_text(s: str, lang="ZH"):
    s2 = normalize_text(s)
    toks = s2.split()
    if lang == "EN":
        toks = [t for t in toks if t.lower() not in STOPWORDS_EN]
    else:
        toks = [t for t in toks if t not in STOPWORDS_ZH and len(t) > 1]
    return toks

def build_boosted_query(user_text, discipline=None, region=None, lang="ZH", extra_constraints=""):
    toks = keywords_from_text(user_text or "", lang=lang)
    key_part = " ".join(dict.fromkeys(toks))  # 去重保序
    disc_part = " ".join(DISCIPLINE_HINTS.get(discipline or "", []))
    region_part = " ".join(REGION_CODES.get(region or "", []))
    cons_part = normalize_text(extra_constraints or "")
    boosted = " ".join([p for p in [key_part, disc_part, region_part, cons_part] if p])

    intent = {
        "sub_tasks": ["确定规范类别", "定位条款号与术语", "提取阈值与适用范围"],
        "keyword_queries": [
            {"task": "规范类别", "keywords": (disc_part or "").split()},
            {"task": "区域标准", "keywords": (region_part or "").split()},
            {"task": "问题要点", "keywords": key_part.split()},
        ],
        "semantic_queries": [user_text.strip()[:200]],
        "fields": ["条款号", "章节", "年份", "实施状态"],
        "rerank_rules": ["章节精确匹配 > 规范名命中 > 年份新近性 > 语义相似度"],
        "citation_format": "（条款号｜规范全称｜年份）",
    }
    return boosted.strip(), intent

def maybe_refine_with_llm(draft_text: str, enable_llm: bool, lang="ZH"):
    if not enable_llm:
        return draft_text, False, None
    try:
        _ = st.secrets["openai"]["api_key"]
    except Exception:
        return draft_text, False, "No OpenAI key detected; skipped polishing."
    try:
        from openai import OpenAI
        client = OpenAI(api_key=st.secrets["openai"]["api_key"])
        sys = "You are a prompt engineer. Keep meaning, compress and clarify for retrieval. Output a single line."
        if lang == "ZH":
            sys = "你是提示词工程专家。在不改变意图前提下压实为便于检索的一行短句。"
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            max_tokens=200,
            messages=[{"role":"system","content":sys},{"role":"user","content":draft_text}],
        )
        out = (resp.choices[0].message.content or "").strip().replace("\n", " ")
        return out or draft_text, True, None
    except Exception as e:
        return draft_text, False, f"LLM polishing failed: {e}"

# ---------------- 单列 UI ----------------
st.subheader("Language / 语言")
lang = st.radio("", ["中文", "English"], horizontal=True, label_visibility="collapsed")
lang_code = "EN" if lang == "English" else "ZH"

st.markdown("### Simple / 简单模式")
user_text = st.text_area(
    "Describe your problem / 描述你的问题",
    height=120,
    placeholder="例：一个照明回路中最多可以接多少盏灯？",
)
col1, col2 = st.columns(2)
discipline = col1.selectbox("Discipline / 专业", list(DISCIPLINE_HINTS.keys()), index=0)
region     = col2.selectbox("Region Codes / 区域标准", list(REGION_CODES.keys()), index=0)
constraints = st.text_input("Optional constraints / 可选约束（建筑类型、房间、人数等）", "")
use_llm = st.toggle("Use LLM to polish (optional) / 使用 LLM 优化（可选）", value=False)

if st.button("🚀 Optimize & Save  |  优化并保存", type="primary", use_container_width=True):
    boosted, intent = build_boosted_query(user_text, discipline, region, lang_code, constraints)
    boosted2, polished, warn = maybe_refine_with_llm(boosted, use_llm, lang_code)
    if warn: st.warning(warn)
    final_query = boosted2.strip() or boosted

    st.markdown("**Optimized query / 优化检索语句**")
    st.code(final_query, language="markdown")

    user_id = st.session_state.get("user", {}).get("id")
    username = st.session_state.get("user", {}).get("username")
    payload = {
        "user_id": user_id,
        "username": username,
        "template_key": "simple_boost",
        "template_title": "检索优化 / Query Boost",
        "variables": {
            "discipline": discipline, "region": region,
            "constraints": constraints, "lang": lang
        },
        "input_text": user_text,
        "output_prompt": json.dumps({"boosted_query": final_query, "intent": intent}, ensure_ascii=False),
        "polished_by_llm": polished,
        "lang": lang_code,
        "page": "prompt_template"
    }
    try:
        rec = insert_prompt_run(payload)
        st.success(f"✅ Saved / 已保存（ID={rec['id']}）")
        st.session_state["boosted_query"] = final_query
    except Exception as e:
        st.error(f"❌ 保存失败：{e}")

# 一键带去 Home（如未打补丁，仍可手动复制）
if st.session_state.get("boosted_query"):
    c1, c2 = st.columns([1,1])
    with c1:
        st.download_button(
            "⬇️ Download .txt",
            data=st.session_state["boosted_query"].encode("utf-8"),
            file_name=f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    with c2:
        if st.button("➡️ Use in Home  |  带去首页检索", use_container_width=True):
            st.session_state["home_query_prefill"] = st.session_state["boosted_query"]
            try:
                st.switch_page("Home.py")
            except Exception:
                st.markdown("[Go to Home / 前往首页](../Home.py)")

# 高级模式（折叠显示 RAG 子任务 JSON）
with st.expander("Advanced / 高级模式（显示结构化 RAG 子任务）", expanded=False):
    if user_text.strip():
        tmp_boost, tmp_intent = build_boosted_query(user_text, discipline, region, lang_code, constraints)
        st.code(json.dumps(tmp_intent, ensure_ascii=False, indent=2), language="json")
    else:
        st.caption("请先在上面输入你的问题。")

# 历史 / 检索（放在页面底部，单列更清爽）
st.markdown("---")
st.markdown("### History / Search  |  历史 / 检索")
# 历史 / 检索
me_only = st.toggle("Only mine / 只看我的", value=True)
kw = st.text_input("Keyword (template/output) / 关键字", "")
limit = st.slider("Rows / 条数", 5, 100, 20, step=5)
me_id = st.session_state.get("user", {}).get("id")

try:
    rows = list_recent(limit=limit, kw=kw.strip() or None, only_me=me_only, me_id=me_id)
    for r in rows:
        with st.container(border=True):
            st.markdown(
                f"**#{r['id']} · {r['template_title']}**  \n"
                f"`{r['template_key']}` · by **{r['username'] or 'anonymous'}** · {r['created_at']:%Y-%m-%d %H:%M:%S}"
            )
            st.code((r['preview'] or '').strip(), language="markdown")
            with st.popover("View full / 查看全文"):
                with get_conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute("SELECT variables, input_text, output_prompt, polished_by_llm, lang FROM app.prompt_runs WHERE id=%s", (r["id"],))
                    full = cur.fetchone()
                if full:
                    st.markdown("**Variables**"); st.code(json.dumps(full["variables"], ensure_ascii=False), language="json")
                    st.markdown("**Input**"); st.code(full["input_text"] or "", language="markdown")
                    st.markdown("**Output (JSON)**"); st.code(full["output_prompt"] or "", language="json")
                    st.caption(f"LLM polished: {'Yes' if full['polished_by_llm'] else 'No'} · Lang: {full.get('lang') or '-'}")

# 🔧 在这里补上 except，和 try 对齐
except Exception as e:
    st.error(f"查询失败：{e}")

st.caption("登录后将自动把记录与账号关联；未登录也可匿名保存。")
