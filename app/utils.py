import fitz


def extract_text(path: str) -> str:

    doc = fitz.open(path)
    return "".join(page.get_text() for page in doc)


def generate_chunks(text: str, chunk_size=500, overlap=100) -> list:

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    start = 0
    chunks = []

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks
