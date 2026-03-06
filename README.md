# GenAI RAG API (FastAPI + Ollama)
End-to-end Retrieval-Augmented Generation (RAG) API using FastAPI, FAISS, and Ollama with PDF document ingestion and semantic search.

This project demonstrates a Retrieval-Augmented Generation (RAG) system built using FastAPI and a local LLM via Ollama.

The system retrieves relevant context using embeddings and generates answers using an LLM.

---

# Architecture

User Question  
↓  
Embedding Generation  
↓  
Vector Similarity Search (FAISS)  
↓  
Retrieve Top Chunks  
↓  
Send Context + Question to LLM  
↓  
Generated Answer

---

# Features

• PDF document ingestion  
• Text chunking with overlap  
• Embedding generation using Ollama  
• FAISS vector similarity search  
• Retrieval-Augmented Generation (RAG)  
• FastAPI REST API for querying documents  
• Local LLM inference (no OpenAI API required)

---

# Tech Stack

Python  
FastAPI  
Ollama (Llama3.1)  
FAISS (Vector Database)  
NumPy  
PyPDF (PDF parsing)  
Embeddings  
Semantic Search

---

# Project Structure

genai-rag-api/

documents/        # PDF files  
rag/              # chunking + pdf loader  
main.py           # FastAPI application  
build_index.py    # build vector index  
test_chunks.py    # test chunking logic  
requirements.txt  
vector.index      # FAISS index  
chunks.json       # stored text chunks  

---

# Run the Project

### 1 Clone Repository
