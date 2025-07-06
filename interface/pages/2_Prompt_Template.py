import streamlit as st
from openai import OpenAI
import os

# Call openai_api_key
openai_api_key = st.session_state.get("openai_api_key", "")

# Title and welcome message
st.set_page_config(
    page_title="Prompt Template",
    page_icon="📐",
    layout="centered"
)

st.title("📐 Prompt Template")
st.caption("Build and test customized prompt templates for construction queries.")

# Initial message
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "📝 Let's start creating your custom prompt templates!"}
    ]

# Sidebar
with st.sidebar:
    # st.header("⚙️ Settings")
    # st.slider("Temperature", 0.0, 1.0, 0.7, step=0.01, key="temperature")
    # st.slider("Top-K Retrieved Chunks", 1, 10, 5, step=1, key="top_k")
    # st.markdown("---")
    st.caption("📐  Explore and customize prompt templates for domain-specific tasks.")
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

# Prompt Form
with st.form("myform"):
    topic_text = st.text_input("Enter prompt:", "")
    submitted = st.form_submit_button("Submit")
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
    elif submitted:
        blog_outline(topic_text)
