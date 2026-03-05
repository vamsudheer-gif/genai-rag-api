from fastapi import FastAPI
from pydantic import BaseModel
from ollama import embeddings, chat
import numpy as np

app = FastAPI()

docs = [
    "RAG stands for Retrieval-Augmented Generation. It uses embeddings and vector search to retrieve relevant context.",
    "Chunking splits documents into smaller parts so retrieval works better.",
    "Temperature controls randomness: 0 is deterministic, higher is more random."
]

class QuestionRequest(BaseModel):
    question: str

def embed(text: str):
    return embeddings(model="llama3.1", prompt=text)["embedding"]

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def retrieve(query: str, top_k: int = 3):
    q = embed(query)
    scores = []
    for doc in docs:
        score = cosine_similarity(q, embed(doc))
        scores.append({"score": score, "text": doc})
    scores.sort(reverse=True, key=lambda x: x["score"])
    return scores[:top_k]

@app.post("/ask")
def ask(req: QuestionRequest):
    results = retrieve(req.question, top_k=3)

    context = "\n".join([r["text"] for r in results])

    prompt = f"""
Answer the question ONLY using the context below.
If the answer is not in the context, say: "I don't know based on the provided context."

Context:
{context}

Question: {req.question}
"""

    response = chat(
        model="llama3.1",
        messages=[
            {"role": "system", "content": "You are a grounded assistant. Expand acronyms correctly."},
            {"role": "user", "content": prompt}
        ],
        options={"temperature": 0}
    )

    return {
        "question": req.question,
        "top_chunks": results,
        "context_used": context,
        "answer": response["message"]["content"]
    }