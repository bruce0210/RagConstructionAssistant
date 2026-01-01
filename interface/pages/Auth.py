# interface/pages/Auth.py
import re
import os
from datetime import datetime, timezone
import hashlib
import time
import json  # 保留：以后可能用到

import streamlit as st
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
import psycopg2
import psycopg2.extras

# === 新增：从 Streamlit 会话里拿请求头，提取客户端 IP ===
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    try:
        # 新版路径
        from streamlit.web.server.server import Server
    except Exception:
        # 旧版兼容
        from streamlit.server.server import Server
except Exception:
    get_script_run_ctx = None
    Server = None

# 可选：获取浏览器 UA（无此包不影响运行）
try:
    from streamlit_user_agent import get_user_agent
except Exception:
    get_user_agent = None

# --------------------------- 页面设置 ---------------------------
st.set_page_config(page_title="Log in / Register", page_icon="🔐", layout="centered")
st.title("🔐 Log in / Register")

# 轻微压缩副标题与表单的间距，去掉空白感
st.markdown("""
<style>
section.main h2, section.main .stSubheader { margin-bottom: 0.5rem !important; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.caption("🔐 Log in & Register.")
    st.markdown(
        """
        <a href="https://github.com/bruce0210/rag_construction_assistant" target="_blank">
            <img src="https://github.com/codespaces/badge.svg" alt="Open in GitHub">
        </a>
        """, unsafe_allow_html=True,
    )

# --------------------------- 数据库连接 -------------------------
def get_conn():
    # 优先从 Streamlit secrets 读取 dsn，其次环境变量 DATABASE_URL
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
        raise RuntimeError("未配置数据库连接。请先设置 [postgres].dsn 或环境变量 DATABASE_URL")
    return psycopg2.connect(dsn)

def ensure_schema():
    ddl = """
    CREATE SCHEMA IF NOT EXISTS app;
    CREATE EXTENSION IF NOT EXISTS pgcrypto;

    CREATE TABLE IF NOT EXISTS app.users (
      id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      username      TEXT UNIQUE NOT NULL CHECK (username ~ '^[A-Za-z][A-Za-z0-9_-]{2,31}$'),
      email         TEXT UNIQUE NOT NULL CHECK (email ~ '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$'),
      password_hash TEXT NOT NULL,
      created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
      is_active     BOOLEAN NOT NULL DEFAULT TRUE
    );

    CREATE TABLE IF NOT EXISTS app.user_devices (
      id           BIGSERIAL PRIMARY KEY,
      user_id      UUID NOT NULL REFERENCES app.users(id) ON DELETE CASCADE,
      device_hash  TEXT NOT NULL,
      user_agent   TEXT,
      created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
      last_seen    TIMESTAMPTZ,
      UNIQUE(user_id, device_hash)
    );

    CREATE TABLE IF NOT EXISTS app.login_events (
      id            BIGSERIAL PRIMARY KEY,
      user_id       UUID NOT NULL REFERENCES app.users(id) ON DELETE CASCADE,
      logged_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
      success       BOOLEAN NOT NULL,
      ip            INET,
      country_code  TEXT,
      region        TEXT,
      city          TEXT,
      device_id     BIGINT REFERENCES app.user_devices(id) ON DELETE SET NULL
    );
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(ddl)
        conn.commit()

ensure_schema()

# -------- 获取公网 IP 与地理信息（后端） --------
def fetch_geo_by_ip(ip: str):
    """给定 IP，做一次地理解析；失败则返回仅含 IP 的 dict"""
    if not ip:
        return {"ip": None, "country": None, "region": None, "city": None}
    try:
        import requests
        j = requests.get(f"https://ipapi.co/{ip}/json/", timeout=3).json()
        return {
            "ip": ip,
            "country": j.get("country_code"),
            "region": j.get("region"),
            "city": j.get("city"),
        }
    except Exception:
        return {"ip": ip, "country": None, "region": None, "city": None}

def fetch_ip_geo():
    """不指定 IP 的查询（得到服务器出口 IP 的地理信息）"""
    try:
        import requests
        j = requests.get("https://ipapi.co/json/", timeout=3).json()
        return {
            "ip": j.get("ip"),
            "country": j.get("country_code"),
            "region": j.get("region"),
            "city": j.get("city"),
        }
    except Exception:
        return {"ip": None, "country": None, "region": None, "city": None}

def get_client_ip_from_headers():
    """从 Streamlit 的 WebSocket 请求头里解析客户端 IP（支持多种代理头）"""
    try:
        if not (get_script_run_ctx and Server):
            return None
        ctx = get_script_run_ctx()
        session_id = ctx.session_id if ctx else None
        srv = Server.get_current() if Server else None
        if not (session_id and srv):
            return None
        # 兼容不同版本的内部结构
        session_info = None
        if hasattr(srv, "_session_info"):
            session_info = srv._session_info.get(session_id)
        if session_info is None and hasattr(srv, "_get_session_info"):
            session_info = srv._get_session_info(session_id)
        if session_info is None:
            return None
        ws = getattr(session_info, "ws", None)
        req = getattr(ws, "request", None) if ws else None
        headers = getattr(req, "headers", {}) if req else {}
        # 各类代理 / CDN 常见头
        xff = headers.get("X-Forwarded-For")
        cfip = headers.get("CF-Connecting-IP")
        xreal = headers.get("X-Real-IP")
        remote_ip = getattr(req, "remote_ip", None)

        # X-Forwarded-For 可能是 "client, proxy1, proxy2"
        if xff:
            ip = xff.split(",")[0].strip()
        else:
            ip = cfip or xreal or remote_ip
        return ip
    except Exception:
        return None

# ------------------------ 校验与工具函数 ------------------------
USERNAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{2,31}$")
RESERVED = {"admin", "root", "system", "null", "none", "select", "insert", "delete", "drop", "table"}

def valid_username(name: str) -> bool:
    return bool(USERNAME_RE.match(name or "")) and name.lower() not in RESERVED

# def valid_email(email: str) -> bool:
#     return bool(re.match(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$", email or ""))
EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
def valid_email(email: str) -> bool:
    return bool(EMAIL_RE.fullmatch((email or "").strip()))


ph = PasswordHasher()  # Argon2id，默认参数安全

def device_fingerprint(ua: str) -> str:
    return hashlib.sha256((ua or "unknown").encode("utf-8")).hexdigest()

def username_exists(username: str) -> bool:
    sql = "SELECT 1 FROM app.users WHERE username=%s"
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, (username,))
        return cur.fetchone() is not None

def email_exists(email: str) -> bool:
    sql = "SELECT 1 FROM app.users WHERE email=%s"
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, (email,))
        return cur.fetchone() is not None

def create_user(username: str, email: str, password_hash: str):
    sql = """
    INSERT INTO app.users (username, email, password_hash)
    VALUES (%s, %s, %s)
    RETURNING id, username, email, created_at
    """
    with get_conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, (username, email, password_hash))
        row = cur.fetchone()
        conn.commit()
        return dict(row)

def fetch_user_by_login(login: str):
    # 允许用用户名或邮箱登录
    sql = """
    SELECT id, username, email, password_hash, created_at
    FROM app.users
    WHERE username=%s OR email=%s
    """
    with get_conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, (login, login))
        row = cur.fetchone()
        return dict(row) if row else None

def log_login_event(user_id, success: bool, ip=None, ua=None, country=None, region=None, city=None):
    device_id = None
    dev_hash = device_fingerprint(ua or "")
    with get_conn() as conn, conn.cursor() as cur:
        # upsert user_devices
        cur.execute(
            """
            INSERT INTO app.user_devices (user_id, device_hash, user_agent)
            VALUES (%s, %s, %s)
            ON CONFLICT (user_id, device_hash) DO UPDATE SET last_seen = now()
            RETURNING id
            """,
            (user_id, dev_hash, ua),
        )
        device_id = cur.fetchone()[0]
        # login_events
        cur.execute(
            """
            INSERT INTO app.login_events (user_id, success, ip, country_code, region, city, device_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (user_id, success, ip, country, region, city, device_id),
        )
        conn.commit()

# ---------------------------- UI ----------------------------
tab_login, tab_register = st.tabs(["Log in", "Register"])

with tab_register:
    st.subheader("Create Account")
    r_username = st.text_input("Username (3-32, starts with a letter, contains only letters/numbers/underscores/hyphens).", key="r_user")
    r_email    = st.text_input("Email", key="r_mail")
    r_pwd      = st.text_input("Password (at least 8 characters)", type="password", key="r_pwd")
    r_pwd2     = st.text_input("Confirm Password", type="password", key="r_pwd2")
    agree      = st.checkbox("I have read and agree to the User Agreement.", value=True, key="r_agree")

    if st.button("Register", type="primary", use_container_width=True, key="btn_register"):
        if not valid_username(r_username):
            st.error("Username format is incorrect, please re-enter.")
        elif not valid_email(r_email):
            st.error("Email format is incorrect.")
        elif r_pwd != r_pwd2 or len(r_pwd) < 8:
            st.error("Passwords do not match, or password length is less than 8 characters.")
        elif username_exists(r_username):
            st.error("This username already exists.")
        elif email_exists(r_email):
            st.error("This email address has already been registered.")
        elif not agree:
            st.error("Please check the box to agree to the User Agreement first.")
        else:
            pwd_hash = ph.hash(r_pwd) # Argon2 加密
            user = create_user(r_username, r_email, pwd_hash)
            st.session_state["user"] = {
                "id": user["id"],
                "username": user["username"],
                "email": user["email"],
            }
            st.success("Registration successful, redirecting to the homepage~")
            st.balloons()
            time.sleep(2)
            try:
                st.switch_page("Home.py") # 直接跳回首页
            except Exception:
                st.markdown("[Return to homepage](../Home.py)") # 兼容旧版

with tab_login:
    st.subheader("Account Login")

    l_login = st.text_input("Username or Email address", key="l_login")
    l_pwd   = st.text_input("Password", type="password", key="l_pwd")
    col1, col2 = st.columns([1,1])
    remember = col1.checkbox("Remember me", value=True, key="l_remember")
    btn = col2.button("Log in", type="primary", use_container_width=True, key="btn_login")

    if btn:
        user = fetch_user_by_login(l_login)

        # --- UA ---
        ua = None
        if get_user_agent:
            info = get_user_agent()
            ua = info.get("user_agent")

        # --- 关键：直接从请求头拿客户端 IP；拿到后用该 IP 做地理解析 ---
        ip = get_client_ip_from_headers()
        if ip:
            geo = fetch_geo_by_ip(ip)
        else:
            # 兜底（少见）：拿不到时退回服务器出口 IP 的地理信息
            geo = fetch_ip_geo()

        ip = geo.get("ip")
        country = geo.get("country")
        region  = geo.get("region")
        city    = geo.get("city")

        if not user:
            st.error("Account does not exist, please register first~")
        else:
            try:
                ph.verify(user["password_hash"], l_pwd)
                # 成功登录：写入 IP/地区
                log_login_event(
                    user["id"], True,
                    ip=ip, ua=ua, country=country, region=region, city=city
                )
                # 会话：供 Home.py 显示登录状态/用户名
                st.session_state["user"] = {
                    "id": user["id"],
                    "username": user["username"],
                    "email": user["email"],
                }
                st.success("Login successful, redirecting to the homepage~")
                st.balloons()
                time.sleep(2)
                try:
                    st.switch_page("Home.py")
                except Exception:
                    st.markdown("[Return to homepage](../Home.py)")
            except VerifyMismatchError:
                # 失败登录也写入（含 IP/地区）
                log_login_event(
                    user["id"], False,
                    ip=ip, ua=ua, country=country, region=region, city=city
                )
                st.error("Incorrect password, please try again.~")

# 底部补充说明（可删除）
st.caption("Note: After successful login, you can see your Username in the sidebar. To log out, please click Logout button.")
