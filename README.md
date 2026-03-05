# GenAI RAG API (FastAPI + Ollama)

This project demonstrates a Retrieval-Augmented Generation (RAG) system built using FastAPI and a local LLM via Ollama.

The system retrieves relevant context using embeddings and generates answers using an LLM.

---

## Architecture

User Question  
↓  
Embedding Generation  
↓  
Vector Similarity Search  
↓  
Retrieve Top Chunks  
↓  
Send Context + Question to LLM  
↓  
Generated Answer  

---

## Tech Stack

Python  
FastAPI  
Ollama (LLM)  
NumPy  
Vector Similarity Search  

---

## Run the Project

Install dependencies: