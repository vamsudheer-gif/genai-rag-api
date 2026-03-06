from fastapi import FastAPI
from pydantic import BaseModel
import json
import numpy as np
import faiss
from ollama import Client

app = FastAPI()

client = Client(host="http://127.0.0.1:11434")

# Load FAISS index
index = faiss.read_index("vector.index")

# Load chunks
with open("chunks.json", "r") as f:
    chunks = json.load(f)


class QuestionRequest(BaseModel):
    question: str


def embed(text: str):
    return client.embeddings(model="llama3.1", prompt=text)["embedding"]

def retrieve(query: str, top_k=3):
    q = np.array([embed(query)], dtype="float32")
    faiss.normalize_L2(q)

    scores, indices = index.search(q, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "score": float(score),
            "text": chunks[idx]["text"],
            "source": chunks[idx]["source"]
        })

    return results


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask")
def ask(req: QuestionRequest):

    results = retrieve(req.question)

    context = "\n\n".join([r["text"] for r in results])

    prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question: {req.question}
"""

    response = client.chat(
        model="llama3.1",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0}
    )

    return {
        "question": req.question,
        "top_chunks": results,
        "context_used": context,
        "answer": response["message"]["content"]
    }

