import streamlit as st
from openai import OpenAI

# Page configuration
st.set_page_config(
    page_title="RAG Construction Assistant",
    page_icon="🏗️",
    layout="wide"
)

# Title and description
st.title("🏗️ RAG Construction Assistant")
st.markdown("A knowledge-augmented assistant for building codes, construction engineering, and regulatory compliance.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar: Settings panel
with st.sidebar:
    st.header("⚙️ Settings")
    openai_key = st.text_input("OpenAI API Key", type="password")
    temperature = st.slider("Response Temperature", 0.0, 1.0, 0.7, 0.05)
    top_k = st.slider("Top‑K Retrieved Chunks", 1, 10, 5)
    st.caption("Adjust response creativity and document chunk retrieval count.")

# Display past conversation messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box for user query
if prompt := st.chat_input("Ask me about construction codes, engineering procedures, or technical guidelines..."):
    # Save user's message to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat bubble
    with st.chat_message("assistant"):
        if not openai_key:
            st.error("Please enter your OpenAI API key in the sidebar.")
        else:
            try:
                # Initialize OpenAI client
                client = OpenAI(api_key=openai_key)

                # Generate streaming response using chat history
                stream = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                    stream=True,
                )

                # Stream response to UI and append to history
                response = st.write_stream(stream)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error: {e}")
