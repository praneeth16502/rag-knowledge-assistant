# [WIP] LLM-Powered Knowledge Assistant (RAG)

A production-grade Retrieval-Augmented Generation (RAG) system that allows users
to query information from multiple documents using a **local LLM**.

## Architecture
PDFs → Chunking → Embeddings → FAISS → Local LLM (Ollama / Llama 3)

## Tech Stack
- FastAPI
- LangChain
- HuggingFace sentence-transformers
- FAISS
- Ollama (Llama 3)

## Features
- Multi-PDF ingestion
- Semantic search with FAISS
- Local inference (no API keys, no billing)
- Source citations for answers
- Hallucination control
- API-first design

## Run Locally
```bash
ollama pull llama3
uvicorn app.main:app --reload
