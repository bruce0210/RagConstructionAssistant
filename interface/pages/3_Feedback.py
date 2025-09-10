# interface/3_Feedback.py
import os
import psycopg2
import psycopg2.extras
import streamlit as st


# ----------------- 页面设置 -----------------
st.set_page_config(
    page_title="User Feedback",
    page_icon="📝",
    layout="centered"
)
st.title("📝 User Feedback")
st.caption("Give feedback to improve the assistant's performance.")

# 初始消息（保留）
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "💬 Please share your feedback to improve."}
    ]

# ----------------- Sidebar（原样保留） -----------------
with st.sidebar:
    # st.header("⚙️ Settings")
    # st.slider("Temperature", 0.0, 1.0, 0.7, step=0.01, key="temperature")
    # st.slider("Top-K Retrieved Chunks", 1, 10, 5, step=1, key="top_k")
    # st.markdown("---")
    st.caption("📝  Provide feedback to improve the assistant's performance and accuracy.")
    st.markdown(
        """
        <a href="https://github.com/bruce0210/rag_construction_assistant" target="_blank">
            <img src="https://github.com/codespaces/badge.svg" alt="Open in GitHub">
        </a>
        """,
        unsafe_allow_html=True,
    )

# ----------------- 数据库连接（与 Auth 保持读取顺序一致） -----------------
def get_conn():
    """
    读取 DSN 优先级：
    1) st.secrets['postgres']['dsn'] 或 st.secrets['DATABASE_URL']
    2) 环境变量 DATABASE_URL
    """
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
    """
    仅创建与反馈直接相关的一张表，最小改动，不引入其他未来结构。
    """
    ddl = """
    CREATE SCHEMA IF NOT EXISTS app;

    CREATE TABLE IF NOT EXISTS app.feedback (
        id           BIGSERIAL PRIMARY KEY,
        user_id      UUID REFERENCES app.users(id) ON DELETE SET NULL,
        message_text TEXT NOT NULL,
        page         TEXT,
        created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(ddl)
        conn.commit()

ensure_schema()

def insert_feedback(message_text: str, user_id=None, page: str = "feedback"):
    sql = """
    INSERT INTO app.feedback (user_id, message_text, page)
    VALUES (%s, %s, %s)
    RETURNING id, created_at
    """
    with get_conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, (user_id, message_text, page))
        row = cur.fetchone()
        conn.commit()
        return dict(row)

# ----------------- 反馈表单 -----------------
with st.form("my_form"):
    text = st.text_area(
        "Enter text:",
        "Please share your suggestions or report any issues encountered. ^_^"
    )
    submitted = st.form_submit_button("Submit")

    if submitted:
        msg = (text or "").strip()
        if not msg:
            st.warning("内容不能为空哦～")
        else:
            try:
                # 若已登录则关联 user_id（保持与 Home/Auth 的用户会话结构一致）
                user_id = None
                if st.session_state.get("user") and st.session_state["user"].get("id"):
                    user_id = st.session_state["user"]["id"]

                rec = insert_feedback(msg, user_id=user_id, page="feedback")
                st.success(f"提交成功，感谢反馈！")
                # st.success(f"提交成功，感谢反馈！已记录（ID={rec['id']}，时间={rec['created_at']}）")
                st.balloons()
                # 可选：清空输入框（刷新当前页）
                st.session_state["last_feedback_ok"] = True
            except Exception as e:
                st.error(f"提交失败：{e}")

# 友好提示
st.caption("提示：若你已登录，我们会将反馈与账号关联；未登录也可匿名提交。感谢你的建议！")
