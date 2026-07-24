from embeddings import generate_embeddings
from utils import extract_text, generate_chunks
from vectorstore import create_or_get_collection, add_documents

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

collection = create_or_get_collection(name="docqa")

database = add_documents(
    collection=collection, embedds=embedds, ids=ids, docs=docs, metadatas=metadatas
)
