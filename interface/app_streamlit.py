# interface/app_streamlit.py

import streamlit as st
from datetime import datetime

# Initialize chat history in session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Page configuration
st.set_page_config(page_title="RAG Construction Assistant", page_icon="🏗️", layout="wide")
st.title("🏗️ RAG Construction Assistant")
st.markdown("A knowledge-aware assistant for construction regulations & engineering Q&A.")

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    temperature = st.slider("Response Creativity", 0.0, 1.0, 0.7, step=0.05)
    top_k = st.slider("Top-K Retrieved Chunks", 1, 10, 5)

# Display chat history
for chat in st.session_state.chat_history:
    role, msg = chat["role"], chat["message"]
    with st.chat_message(role):
        st.markdown(msg)

# Input field for user
prompt = st.chat_input("Ask about building codes, engineering rules, or construction workflows...")

# Response logic
if prompt:
    # Display user input
    st.chat_message("user").markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "message": prompt})

    # Simulate assistant response (to be replaced with actual backend logic)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving documents and generating response..."):
            # Placeholder response
            response = f"🔍 You asked: **{prompt}**\n\n_(This is a placeholder response. Actual RAG pipeline not yet connected.)_"
            st.markdown(response)

    # Save assistant response to history
    st.session_state.chat_history.append({"role": "assistant", "message": response})
