from src.config import settings
from src.llm_engine.ollama_client import OllamaWrapper
from src.llm_engine.base import BaseLLM


def get_llm() -> BaseLLM:
    if settings.LLM_Provider == "ollama":
        return OllamaWrapper()
    
    raise ValueError(f"Unsupported LLM Provider: {settings.LLM_Provider}")

    