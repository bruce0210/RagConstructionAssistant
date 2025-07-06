import streamlit as st
from openai import OpenAI
import os

## Call openai_api_key
openai_api_key = st.session_state.get("openai_api_key", "")

# Title and welcome message
st.set_page_config(
    page_title="User Feedback",
    page_icon="📝",
    layout="centered"
)

st.title("📝 User Feedback")
st.caption("Give feedback to improve the assistant's performance.")

# Initial message
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "💬 Please share your feedback to improve."}
    ]

# Sidebar
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

# Display chat history
# for msg in st.session_state.messages:
    # st.chat_message(msg["role"]).write(msg["content"])

# Feedback Form
with st.form("my_form"):
    text = st.text_area("Enter text:", "Please share your suggestions or report any issues encountered. ^_^")
    submitted = st.form_submit_button("Submit")
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
    elif submitted:
        generate_response(text)
