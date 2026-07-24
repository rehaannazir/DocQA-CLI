import chromadb
from helper import extract_text, generate_chunks, generate_embeddings

sources = [
    "docs/container_orchester.pdf",
    "docs/high_traffic_apis.pdf",
    "docs/message_queues.pdf",
]

docs = []
metadatas = []
ids = []
embedds = []

for source in sources:

    text = extract_text(path=source)
    chunks = generate_chunks(text=text)

    for i, chunk in enumerate(chunks):

        docs.append(chunk)
        metadatas.append({"source": source, "index": i})
        ids.append(f"{source}_{i}")

embedds = generate_embeddings(docs)

client = chromadb.PersistentClient("data/")

try:
    client.delete_collection(name="docqa")
except (ValueError, chromadb.errors.NotFoundError):
    pass

collection = client.get_or_create_collection(
    name="docqa", metadata={"hnsw:space": "cosine"}
)

database = collection.add(
    embeddings=embedds, ids=ids, documents=docs, metadatas=metadatas
)
