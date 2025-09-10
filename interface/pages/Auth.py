# interface/pages/Auth.py
import re
import os
from datetime import datetime, timezone
import hashlib
import time

import streamlit as st
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
import psycopg2
import psycopg2.extras

# 可选：获取浏览器 UA（无此包不影响运行）
try:
    from streamlit_user_agent import get_user_agent
except Exception:
    get_user_agent = None

# --------------------------- 页面设置 ---------------------------
st.set_page_config(page_title="登录 / 注册", page_icon="🔐", layout="centered")
st.title("🔐 登录 / 注册")

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

# ------------------------ 校验与工具函数 ------------------------
USERNAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{2,31}$")
RESERVED = {"admin", "root", "system", "null", "none", "select", "insert", "delete", "drop", "table"}

def valid_username(name: str) -> bool:
    return bool(USERNAME_RE.match(name or "")) and name.lower() not in RESERVED

def valid_email(email: str) -> bool:
    return bool(re.match(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$", email or ""))

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
tab_login, tab_register = st.tabs(["登录", "注册"])

with tab_register:
    st.subheader("创建账户")
    r_username = st.text_input("用户名（3-32，字母开始，仅含字母/数字/下划线/连字符）", key="r_user")
    r_email    = st.text_input("邮箱", key="r_mail")
    r_pwd      = st.text_input("密码（至少 8 位）", type="password", key="r_pwd")
    r_pwd2     = st.text_input("重复密码", type="password", key="r_pwd2")
    agree      = st.checkbox("我已阅读并同意《用户协议》", value=True, key="r_agree")

    if st.button("注册", type="primary", use_container_width=True, key="btn_register"):
        if not valid_username(r_username):
            st.error("用户名格式不正确，请重新输入。")
        elif not valid_email(r_email):
            st.error("邮箱格式不正确。")
        elif r_pwd != r_pwd2 or len(r_pwd) < 8:
            st.error("两次密码不一致，或密码长度不足 8 位。")
        elif username_exists(r_username):
            st.error("该用户名已存在。")
        elif email_exists(r_email):
            st.error("该邮箱已被注册。")
        elif not agree:
            st.error("请先勾选同意《用户协议》。")
        else:
            pwd_hash = ph.hash(r_pwd) # Argon2 加密
            user = create_user(r_username, r_email, pwd_hash)
            st.session_state["user"] = {
                "id": user["id"],
                "username": user["username"],
                "email": user["email"],
            }
            st.success("注册成功，正在跳转到首页…")
            st.balloons()
            time.sleep(2)
            try:
                st.switch_page("Home.py") # 直接跳回首页
            except Exception:
                st.markdown("[返回首页](../Home.py)") # 兼容旧版

with tab_login:
    st.subheader("账户登录")
    l_login = st.text_input("用户名或邮箱", key="l_login")
    l_pwd   = st.text_input("密码", type="password", key="l_pwd")
    col1, col2 = st.columns([1,1])
    remember = col1.checkbox("记住我", value=True, key="l_remember")
    btn = col2.button("登录", type="primary", use_container_width=True, key="btn_login")

    if btn:
        user = fetch_user_by_login(l_login)
        ip = None
        ua = None
        if get_user_agent:
            info = get_user_agent()
            ua = info.get("user_agent")

        if not user:
            st.error("账户不存在。")
        else:
            try:
                ph.verify(user["password_hash"], l_pwd)
                log_login_event(user["id"], True, ip=ip, ua=ua)
                # 会话：供 Home.py 读取显示登录状态/用户名
                st.session_state["user"] = {
                    "id": user["id"],
                    "username": user["username"],
                    "email": user["email"],
                }
                st.success("登录成功，正在跳转到首页…")
                st.balloons()
                time.sleep(2)
                try:
                    st.switch_page("Home.py")           # 新增：直接跳回首页
                except Exception:
                    st.markdown("[返回首页](../Home.py)")  # 兼容旧版
            except VerifyMismatchError:
                log_login_event(user["id"], False, ip=ip, ua=ua)
                st.error("密码错误。")

# 底部补充说明（可删除）
st.caption("提示：登录成功后，可在侧栏看到“已登录：用户名”。如需退出，请在侧栏点击“退出登录”。")
