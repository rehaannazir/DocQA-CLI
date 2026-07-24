import tenacity
import logging
from google import genai
from tenacity import retry
from google.genai import types, errors
from config import gemini_api_key, chat_model
from log import get_logger

client = genai.Client(api_key=gemini_api_key)

logger = get_logger(__name__)


class retryable_api_call(Exception):
    pass


class non_retryable_api_call(Exception):
    pass


def check_status(response):

    if response in [500, 502, 503, 504, 429]:

        raise retryable_api_call(
            f"RetryableAPICall Error Occur. Status code {response}"
        )

    if 400 < response < 500:

        raise non_retryable_api_call(
            f"NonRetryableAPICall Error Occur. Status code {response}"
        )


@retry(
    retry=tenacity.retry_if_exception_type(
        (ConnectionError, TimeoutError, retryable_api_call)
    ),
    wait=tenacity.wait_exponential(multiplier=1, min=2, max=4),
    stop=tenacity.stop_after_attempt(2),
    before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def generate_answer(prompt):

    logger.info("Calling chat model %s", chat_model)

    try:
        stream = client.models.generate_content_stream(
            model=chat_model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.1),
        )
        chunks = list(stream)

        logger.info("Received response from chat model %s", chat_model)

        return chunks

    except errors.APIError as e:
        check_status(e.code)
        raise
