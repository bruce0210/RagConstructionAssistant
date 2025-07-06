import streamlit as st
from openai import OpenAI
import os

# Call openai_api_key
openai_api_key = st.session_state.get("openai_api_key", "")

# Title and welcome message
st.set_page_config(
    page_title="File Upload",
    page_icon="📄",
    layout="centered"
)

st.title("📄 File Upload")
st.caption("Upload construction-related documents.")

# Initial message
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "📎 Please upload a file to update model training."}
    ]

# Sidebar
with st.sidebar:
   # st.header("⚙️ Settings")
   # st.slider("Temperature", 0.0, 1.0, 0.7, step=0.01, key="temperature")
   # st.slider("Top-K Retrieved Chunks", 1, 10, 5, step=1, key="top_k")
   # st.markdown("---")
    st.caption("📄  Upload construction-related documents.")
    st.markdown(
        """
        <a href="https://github.com/bruce0210/rag_construction_assistant" target="_blank">
            <img src="https://github.com/codespaces/badge.svg" alt="Open in GitHub">
        </a>
        """,
        unsafe_allow_html=True,
    )

# Upload file
uploaded_file = st.file_uploader("📎 Upload your PDF or DOCX file", type=["pdf", "docx"])

if uploaded_file:
    st.success(f"File '{uploaded_file.name}' uploaded successfully.")
    # You can implement file processing and question answering here

# Display chat history
# for msg in st.session_state.messages:
   # st.chat_message(msg["role"]).write(msg["content"])

