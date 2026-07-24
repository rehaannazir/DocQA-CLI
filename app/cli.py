import time
from ingest import run_ingestion, DEFAULT_SOURCES
from vectorstore import create_or_get_collection, similarity_search
from embeddings import generate_embedding
from llm import generate_answer, retryable_api_call, non_retryable_api_call
from log import get_logger

logger = get_logger(__name__)


def build_prompt(question, context_chunks):

    context = "\n\n".join(context_chunks)

    return (
        "Answer the question using only the context below. "
        "If the answer isn't in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )


def ask(question, collection):

    query_embedding = generate_embedding(query=question)
    results = similarity_search(collection, query_embedding)

    prompt = build_prompt(question, results["documents"][0])
    try:
        response = generate_answer(prompt)

        for chunk in response:
            print(chunk.text, end="", flush=True)
            time.sleep(0.02)

    except (retryable_api_call, non_retryable_api_call) as e:
        logger.error("API call failed: %s", e)

    except Exception:
        logger.exception("Unexpected error during call_client")

    print()


def main():

    collection = create_or_get_collection(name="docqa")

    if collection.count() == 0:
        collection = run_ingestion(DEFAULT_SOURCES)

    while True:

        try:
            question = input("\nAsk question(or 'exit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if question.lower() in ("exit", "quit"):
            break

        if not question:
            continue

        ask(question, collection)


if __name__ == "__main__":
    main()
