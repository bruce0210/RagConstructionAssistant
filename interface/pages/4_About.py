# About.py
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="About | RAG Construction Assistant",
    page_icon="ℹ️",
    layout="wide"
)

# ---------------------------
# Small style tweaks
# ---------------------------
st.markdown("""
<style>
.reportview-container .main .block-container{padding-top:1.5rem; padding-bottom:1.2rem;}
table thead tr th {text-align:left !important; font-weight:700;}
table tbody tr td {vertical-align:top;}
.card{border:1px solid rgba(255,255,255,0.08); border-radius:14px; padding:1.0rem 1.2rem; background:rgba(127,127,127,0.06)}
.btnrow a{margin-right:.5rem; margin-bottom:.5rem;}
.footer {opacity:.7; font-size:0.9rem; margin-top:2rem; border-top:1px solid rgba(255,255,255,0.08); padding-top:.8rem;}
.small {font-size:0.92rem;}
.kbd {padding: 0 .3rem; border: 1px solid rgba(127,127,127,.4); border-bottom-width:2px; border-radius:6px; font-size:0.85em;}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Header
# ---------------------------
st.title("ℹ️ About RAG Construction Assistant")
st.caption("A prototype system for construction engineering knowledge management based on artificial intelligence-enhanced retrieval, interpretation, and evaluation of building standards.")

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    # st.markdown("---")
    st.caption("ℹ️ A prototype system for construction engineering knowledge management based on artificial intelligence-enhanced retrieval, interpretation, and evaluation of building standards.")
    st.markdown(
        """
        <a href="https://github.com/bruce0210/rag_construction_assistant" target="_blank">
            <img src="https://github.com/codespaces/badge.svg" alt="Open in GitHub">
        </a>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------
# Overview + Quick facts
# ---------------------------
col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("Overview")
    st.markdown(
        """
**RAG Construction Assistant** is a prototype system for retrieval and interpretation of **construction standards** (GB, IFC, etc.).  

It integrates **Retrieval-Augmented Generation (RAG)** with domain ontology, re-ranking, and multi-agent interaction.  
The system ensures that users always receive the **original standard clause first**, followed by **AI-assisted explanations, comparisons, and application insights**.

Main highlights:
- **Reproducible indexing pipeline** (DOCX/PDF → segmentation → embeddings → FAISS)  
- **Two-stage retrieval**: semantic vector recall + GPT-4o/4o-mini re-ranking  
- **Role-based agent interface**: planner / reviewer / contractor perspectives  
- **Evaluation-ready**: designed for academic research and reproducibility  
"""
    )

with col2:
    st.subheader("At a Glance")
    st.markdown(
        """
- **Stack**: Python · Streamlit · FAISS · GPT-4o/4o-mini  
- **Corpus**: curated DOCX/PDF construction standards (with tables, figures, notes)  
- **Focus**: retrieval accuracy, first-hit rate, latency, expert rating  
- **Use cases**: clause lookup, interpretation, terminology alignment, Q&A  
"""
    )
    st.markdown('<div class="btnrow">', unsafe_allow_html=True)
    st.link_button("⭐ GitHub Repository", "https://github.com/bruce0210/RagConstructionAssistant")
    st.link_button("⬢ Open in Codespaces", "https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=867248032")
    st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# ---------------------------
# System Modules
# ---------------------------
st.subheader("System Modules")

modules_df = pd.DataFrame([
    {"Module": "📚 Ontology Engine", "Description": "Structures construction terms, roles, codes, and workflows."},
    {"Module": "🧠 Retrieval Core", "Description": "Embedding + FAISS retrieval with GPT-4o/4o-mini re-ranking."},
    {"Module": "🗣️ Agent Interface", "Description": "Multi-agent frontend simulating planner / reviewer / contractor."},
    {"Module": "🖥️ Frontend UI", "Description": "Streamlit-based interface for interaction & evaluation."},
], columns=["Module", "Description"])

st.table(modules_df)

# ---------------------------
# Key Features
# ---------------------------
st.subheader("Key Features")
st.markdown(
    """
- **Original text first**: always display the **exact clause text** before AI explanations.  
- **Figures / notes parsing**: handles tables, images, and “Explanatory Notes” and links them with their parent clauses.  
- **Two-stage retrieval**: vector similarity recall + GPT-4o/4o-mini re-ranking.  
- **Evaluation metrics**: average retrieval success, first-hit accuracy, response time, expert ratings.  
- **Modular design**: easy to swap embeddings/LLMs, extend ontology, and add agent roles.  
"""
)

# ---------------------------
# Technical Notes
# ---------------------------
with st.expander("Technical Notes"):
    st.markdown(
        """
- **Indexing pipeline**: clause segmentation → text cleaning → embedding → FAISS index (with metadata such as doc name, clause ID, linked media).  
- **Query pipeline**: user query → embedding → FAISS recall → GPT-4o/4o-mini re-ranking → prompt construction → response.  
- **Frontend**: Streamlit multipage app with adjustable interpretation depth, similarity threshold filter, and modal popups.  
"""
    )

st.divider()

# ---------------------------
# Project Team
# ---------------------------
st.subheader("👥 Project Team")

left, right = st.columns([0.8, 2.5])

with left:
    avatar_url = "https://ragca-project-attachments.oss-ap-northeast-1.aliyuncs.com/bruce.jpg"
    st.image(avatar_url, width=160)
with right:
    st.markdown(
        """
**Hao Yin** · [hi-bruce.yin@connect.polyu.hk](mailto:hi-bruce.yin@connect.polyu.hk)  
*MSc Student · Researcher · Developer*  
**Master in Information Systems**  
**COMP | The Hong Kong Polytechnic University**

Research focus: **AI-enhanced construction knowledge management**,  
with interests in **RAG**, **agent systems**, and **semantic retrieval**.  
Responsible for **system architecture, development, and evaluation**.
""")

st.divider()

# ---------------------------
# Resources & Links
# ---------------------------
st.subheader("Resources & Links")
st.markdown(
    """
- 📘 **Repository**: <https://github.com/bruce0210/RagConstructionAssistant>  
- 🧪 **Issues & Roadmap**: GitHub Issues for tracking features, bugs, and experiment logs.  
- 📝 **Research Context**: aligns with MSc thesis on *AI-Enhanced Construction Information Flow Management with Version Control*,  
  focusing on **knowledge augmentation in RAG**, **dual retrieval + re-ranking**, and **evaluation framework design**.  
"""
)

# ---------------------------
# Footer
# ---------------------------
st.markdown(
    """
<div class="footer">
© 2025 RAG Construction Assistant. All rights reserved.  
</div>
""",
    unsafe_allow_html=True
)
