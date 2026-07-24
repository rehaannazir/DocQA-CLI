from google import genai
from google.genai import types
from config import embedding_model, gemini_api_key
from log import get_logger

client = genai.Client(api_key=gemini_api_key)

logger = get_logger(__name__)


def get_embedding_model():

    return embedding_model


def generate_embeddings(chunks: list) -> list:

    logger.info("Generating embeddings for %d chunks using model %s", len(chunks), embedding_model)

    embedds = []

    for chunk in chunks:

        response = client.models.embed_content(
            model=embedding_model,
            contents=chunk,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )

        embedds.append(response.embeddings[0].values)

    logger.info("Generated %d embeddings", len(embedds))

    return embedds


def generate_embedding(query: str) -> list:

    logger.info("Generating embedding for query using model %s", embedding_model)

    response = client.models.embed_content(
        model=embedding_model,
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )

    logger.info("Generated embedding for query")

    return response.embeddings[0].values
