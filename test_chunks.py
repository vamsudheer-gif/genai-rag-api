from rag.pdf_loader import load_pdf
from rag.chunker import chunk_text

# Load PDF
text = load_pdf("documents/sample.pdf")

print("Total characters:", len(text))

# Chunk the text
chunks = chunk_text(text)

print("Total chunks:", len(chunks))

print("\nFirst chunk preview:\n")
print(chunks[0][:500])