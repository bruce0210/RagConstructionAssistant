# core/telemetry.py
import os, json
from typing import Optional, Dict, Any, List
import psycopg2
import psycopg2.extras

def _get_conn():
    # 支持 Streamlit secrets 与环境变量
    dsn = None
    try:
        import streamlit as st  # 允许在非 Streamlit 环境下导入
        try:
            if "postgres" in st.secrets and "dsn" in st.secrets["postgres"]:
                dsn = st.secrets["postgres"]["dsn"]
            elif "DATABASE_URL" in st.secrets:
                dsn = st.secrets["DATABASE_URL"]
        except Exception:
            pass
    except Exception:
        pass
    if not dsn:
        dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("未配置数据库连接。请设置 [postgres].dsn 或环境变量 DATABASE_URL")
    return psycopg2.connect(dsn)

def ensure_schema():
    """
    仅创建“首页查询落库”所需最小三表：
    - app.search_queries     : 记录一次‘搜’（含模型/TopK/耗时/Top-K 条款编号列表）
    - app.search_answers     : 记录 LLM 解读文本（与某次 query 关联）
    - app.answer_reactions   : 记录用户对某条 answer 的点赞/点踩
    另：app.hit_reactions    : 记录用户对 Top-K 命中条款（以条款号识别）的相关性反馈
    """
    ddl = """
    CREATE SCHEMA IF NOT EXISTS app;

    CREATE TABLE IF NOT EXISTS app.search_queries (
        id          BIGSERIAL PRIMARY KEY,
        user_id     UUID REFERENCES app.users(id) ON DELETE SET NULL,
        query_text  TEXT NOT NULL,
        topk        INT,
        model_name  TEXT,
        page        TEXT,
        started_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
        finished_at TIMESTAMPTZ,
        latency_ms  INTEGER,
        n_hits      INT,
        top_hits    JSONB   -- 仅存命中条款编号数组，不存任何原文
    );

    CREATE TABLE IF NOT EXISTS app.search_answers (
        id           BIGSERIAL PRIMARY KEY,
        query_id     BIGINT NOT NULL REFERENCES app.search_queries(id) ON DELETE CASCADE,
        answer_text  TEXT NOT NULL,
        evidence     JSONB,  -- 这里也不放全文，只放必要摘要
        created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
    );

    CREATE TABLE IF NOT EXISTS app.answer_reactions (
        id         BIGSERIAL PRIMARY KEY,
        answer_id  BIGINT NOT NULL REFERENCES app.search_answers(id) ON DELETE CASCADE,
        user_id    UUID REFERENCES app.users(id) ON DELETE SET NULL,
        reaction   SMALLINT NOT NULL CHECK (reaction IN (-1, 1)),
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        UNIQUE (answer_id, user_id)
    );

    -- 每个 Top-K 命中的条款相关性反馈（只按条款号记录，不存原文）
    CREATE TABLE IF NOT EXISTS app.hit_reactions (
        id         BIGSERIAL PRIMARY KEY,
        query_id   BIGINT NOT NULL REFERENCES app.search_queries(id) ON DELETE CASCADE,
        clause_no  TEXT   NOT NULL,
        user_id    UUID REFERENCES app.users(id) ON DELETE SET NULL,
        reaction   SMALLINT NOT NULL CHECK (reaction IN (-1, 1)),
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        UNIQUE (query_id, clause_no, user_id)
    );
    """
    # 兼容已建表后新增 top_hits 字段
    alter = "ALTER TABLE app.search_queries ADD COLUMN IF NOT EXISTS top_hits JSONB;"
    with _get_conn() as conn, conn.cursor() as cur:
        cur.execute(ddl)
        cur.execute(alter)
        conn.commit()

def start_query(user_id: Optional[str], query_text: str, topk: int, model_name: str,
                page: str = "home") -> int:
    sql = """
    INSERT INTO app.search_queries (user_id, query_text, topk, model_name, page)
    VALUES (%s, %s, %s, %s, %s)
    RETURNING id
    """
    with _get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, (user_id, query_text, topk, model_name, page))
        qid = cur.fetchone()[0]
        conn.commit()
        return qid

def finish_query(query_id: int, n_hits: int, latency_ms: Optional[int],
                 top_clause_nos: Optional[List[str]] = None) -> None:
    """
    top_clause_nos: 仅传 Top-K 的条款编号列表（如 ["6.4.3", "5.2.1", ...]），不存任何原文。
    """
    sql = """
    UPDATE app.search_queries
       SET finished_at = now(),
           n_hits = %s,
           latency_ms = %s,
           top_hits = %s
     WHERE id = %s
    """
    top_hits_json = json.dumps(top_clause_nos or [], ensure_ascii=False)
    with _get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, (n_hits, latency_ms, top_hits_json, query_id))
        conn.commit()

def save_answer(query_id: int, answer_text: str, evidence: Optional[Dict[str, Any]] = None) -> int:
    """
    evidence 也避免存全文；只放必要的编号/分数等元数据。
    """
    ev_json = json.dumps(evidence, ensure_ascii=False) if evidence is not None else None
    sql = """
    INSERT INTO app.search_answers (query_id, answer_text, evidence)
    VALUES (%s, %s, %s)
    RETURNING id
    """
    with _get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, (query_id, answer_text, ev_json))
        aid = cur.fetchone()[0]
        conn.commit()
        return aid

def set_reaction(answer_id: int, user_id: Optional[str], reaction: int) -> None:
    """
    reaction: 1=点赞；-1=点踩；同一用户对同一 answer 再次操作会覆盖（upsert）。
    """
    sql = """
    INSERT INTO app.answer_reactions (answer_id, user_id, reaction)
    VALUES (%s, %s, %s)
    ON CONFLICT (answer_id, user_id)
      DO UPDATE SET reaction = EXCLUDED.reaction,
                    created_at = now()
    """
    with _get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, (answer_id, user_id, reaction))
        conn.commit()

def set_hit_reaction(query_id: int, clause_no: str, user_id: Optional[str], reaction: int) -> None:
    """
    记录每个 Top-K 命中条款的“相关/不相关”反馈（只用条款号识别，不存原文）。
    reaction: 1=相关；-1=不相关；同一用户对同一 (query_id, clause_no) 会覆盖（upsert）。
    """
    sql = """
    INSERT INTO app.hit_reactions (query_id, clause_no, user_id, reaction)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (query_id, clause_no, user_id)
      DO UPDATE SET reaction = EXCLUDED.reaction,
                    created_at = now()
    """
    with _get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, (query_id, clause_no, user_id, reaction))
        conn.commit()
