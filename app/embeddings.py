from google import genai
from google.genai import types
from config import embedding_model, gemini_api_key

client = genai.Client(gemini_api_key)


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
