import chromadb
from config import chroma_db_path

client = chromadb.PersistentClient(chroma_db_path)


def get_vectorstore():

    return client


def create_or_get_collection(name, distance_function="cosine"):

    return client.get_or_create_collection(
        name=name, metadata={"hnsw:space": distance_function}
    )


def add_documents(collection, embedds, ids, docs, metadatas):

    return collection.add(
        embeddings=embedds, ids=ids, documents=docs, metadatas=metadatas
    )


def similarity_search(collection, query_embeddings, top_results=3):

    return collection.query(query_embeddings=[query_embeddings], n_results=top_results)
