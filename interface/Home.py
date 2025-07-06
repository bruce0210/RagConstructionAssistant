import streamlit as st
from openai import OpenAI
import os


# Input OPENAI_API_KEY
openai_api_key = os.getenv("OPENAI_API_KEY", "")

# Title and welcome message
st.set_page_config(
    page_title="RAG Construction Assistant",
    page_icon="🏗️",
    layout="centered"  # Avoid default 'wide' view
)

st.title("🏗️ RAG Construction Assistant")
st.caption("Ask questions about building specifications, engineering standards or any construction engineering regulations.")


# Initial message
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "👷‍♂️How can I help you with your construction project today?"}
    ]

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    # openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")

    st.slider("Temperature", 0.0, 1.0, 0.7, step=0.01, key="temperature")
    st.slider("Top-K Retrieved Chunks", 1, 10, 5, step=1, key="top_k")
    st.markdown("---")
    st.caption("🏗️  Ask questions about building specifications, engineering standards or any construction engineering regulations.")
    # st.markdown(
    #     "[![Open in GitHub](https://github.com/codespaces/badge.svg)]"
    #     "(https://github.com/bruce0210/rag_construction_assistant)"
    # )
    st.markdown(
        """
        <a href="https://github.com/bruce0210/rag_construction_assistant" target="_blank">
            <img src="https://github.com/codespaces/badge.svg" alt="Open in GitHub">
        </a>
        """,
        unsafe_allow_html=True,
    )

# Display history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Input interaction
if prompt := st.chat_input("Ask me anything about construction knowledge..."):
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if not openai_api_key:
        st.warning("🔑 Please add your OpenAI API key to continue.")
    else:
        client = OpenAI(api_key=openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=st.session_state.messages,
            temperature=st.session_state.temperature,
        )
        msg = response.choices[0].message.content
        st.chat_message("assistant").write(msg)
        st.session_state.messages.append({"role": "assistant", "content": msg})
