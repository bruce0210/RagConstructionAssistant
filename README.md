# RAG Construction Assistant

A semantic knowledge retrieval system for the construction industry, combining:

- 🧠 LLM-powered question answering
- 🔍 RAG-based retrieval using FAISS + OpenAI embeddings
- 👷 Multi-agent simulation for construction roles
- 📑 Docx-to-ontology parsing for Chinese building regulations

## 📁 Project Structure

- `core/`: Embedding, FAISS indexing, prompt engineering, GPT interface
- `agents/`: Planner, reviewer, contractor agents
- `interface/`: CLI / Streamlit-based simulation
- `data/`: Raw .docx, processed segments, embedding vectors

## 🚀 Quick Start

```bash
conda activate rag_construction_assistant
python run_pipeline.py
```

## 🧠 Author

**Bruce Yin**

PolyU MSc in Information Systems
GitHub: [@bruce0210](https://github.com/bruce0210)
