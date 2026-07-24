import os
from dotenv import load_dotenv

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
chroma_db_path = os.getenv("CHROMADB_PATH")
embedding_model = os.getenv("EMBEDDING_MODEL")
chat_model = os.getenv("CHAT_MODEL")
