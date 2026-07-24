from google import genai
from google.genai import types
from config import embedding_model, gemini_api_key

client = genai.Client(api_key=gemini_api_key)


def get_embedding_model():

    return embedding_model


def generate_embeddings(chunks: list) -> list:

    embedds = []

    for chunk in chunks:

        response = client.models.embed_content(
            model=embedding_model,
            contents=chunk,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )

        embedds.append(response.embeddings[0].values)

    return embedds


def generate_embedding(query: str) -> list:

    response = client.models.embed_content(
        model=embedding_model,
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )

    return response.embeddings[0].values
