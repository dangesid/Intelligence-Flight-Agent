import os 
from dotenv import load_dotenv

load_dotenv()

class Settings:
    LLM_Provider: str = os.getenv("LLM Provider","ollama")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Vector DB
    VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH", "./vector_db")

    # Embeddings
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

settings = Settings()
