def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150):
    """
    Splits text into overlapping chunks (character-based).
    - chunk_size: how big each chunk is (characters)
    - overlap: repeated characters between chunks (helps context)
    """
    text = text.replace("\r", "")
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap
        if start < 0:
            start = 0

        if end == len(text):
            break

    return chunks