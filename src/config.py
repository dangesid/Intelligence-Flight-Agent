import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # ===============================
    # LLM Configuration
    # ===============================

    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "azure")

    AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    AZURE_OPENAI_API_VERSION: str = os.getenv(
        "AZURE_OPENAI_API_VERSION",
        "2025-01-01-preview"
    )

    # ===============================
    # Vector DB Paths
    # ===============================

    VECTOR_DB_PATH_COSMOS: str = os.getenv(
        "VECTOR_DB_PATH_COSMOS_DB",
        "./cosmos_vector_db"
    )

    VECTOR_DB_PATH_CSV: str = os.getenv(
        "VECTOR_DB_PATH_CSV",
        "./csv_vector_db"
    )

    # ===============================
    # Embedding Model
    # ===============================

    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2"
    )


settings = Settings()
