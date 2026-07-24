from google import genai
from google.genai import types
from config import gemini_api_key, chat_model

client = genai.Client(api_key=gemini_api_key)


def generate_answer(prompt):

    response = client.models.generate_content_stream(
        model=chat_model,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.1),
    )

    return response
