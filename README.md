# TEYZIX AI-2 Domain Track - Retrieval Chatbot (Task 2)

An internal knowledge base RAG (Retrieval-Augmented Generation) application designed to accurately parse, index, and query the TEYZIX Internship Guidelines using local AI compute architecture.

## 🏗️ System Architecture
1. **Document Ingestion (`ingestion.py`)**: Extracts policy text using `PyPDFLoader` and segments it into balanced chunks using `RecursiveCharacterTextSplitter`.
2. **Vector Space Mapping**: Transforms raw text blocks into 384-dimensional semantic vectors using the modern `langchain-huggingface` architecture (`all-MiniLM-L6-v2`).
3. **Local Vector Database Storage**: Indexes and saves embeddings locally onto disk storage using a `FAISS` structure matrix.
4. **Context Retrieval**: Runs proximity searches matching user text against index arrays to retrieve top context elements (`k=3`).
5. **Local Inference Execution**: Feeds context strings into a locally configured `llama3` instance via **Ollama** using strict system prompt isolation flags.
6. **Frontend Framework**: Drives UI states and historical chat memories (`st.session_state`) through a web interface built with `Streamlit`.

## 🚀 Setup & Execution Guide

### 1. Prerequisite Installations
Ensure you have **Ollama** running locally with the target model:
```bash
ollama run llama3