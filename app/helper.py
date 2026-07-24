import os
import fitz
from google import genai
from dotenv import load_dotenv
from google.genai import types

load_dotenv()
client = genai.Client(os.getenv("GEMINI_API_KEY"))


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


def generate_embeddings(chunks: list, model="gemini-embedding-001") -> list:

    embedds = []

    for chunk in chunks:

        response = client.models.embed_content(
            model=model,
            contents=chunk,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )

        embedds.append(response.embeddings[0].values)

    return embedds
