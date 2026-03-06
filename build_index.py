import os
from rag.pdf_loader import load_pdf
from rag.chunker import chunk_text
from ollama import embeddings
import numpy as np
import faiss
import json

documents_path = "documents"

all_chunks = []

# load every pdf in documents folder
for file in os.listdir(documents_path):
    if file.endswith(".pdf"):
        pdf_path = os.path.join(documents_path, file)
        print(f"Loading {pdf_path}")

        text = load_pdf(pdf_path)
        chunks = chunk_text(text)

        for chunk in chunks:
            all_chunks.append({
                "text": chunk,
                "source": file
            })

def embed(text):
    return embeddings(model="llama3.1", prompt=text)["embedding"]

vectors = [embed(chunk["text"]) for chunk in all_chunks]
vectors = np.array(vectors).astype("float32")

# normalize vectors
faiss.normalize_L2(vectors)

# build FAISS index
dimension = len(vectors[0])
index = faiss.IndexFlatL2(dimension)
index.add(vectors)

# save index
faiss.write_index(index, "vector.index")

# save chunks
with open("chunks.json", "w") as f:
    json.dump(all_chunks, f)

print("FAISS index created successfully!")
print("Total chunks indexed:", len(all_chunks))
print("Embedding dimension:", dimension)