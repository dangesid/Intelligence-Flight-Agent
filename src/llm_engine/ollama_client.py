from typing import List
from langchain_community.llms import Ollama
from src.llm_engine.base import BaseLLM
from src.config_ollama import settings
import subprocess

class OllamaWrapper(BaseLLM):
    def __init__(self):
        self.llm = Ollama(model=settings.OLLAMA_MODEL, base_url=settings.OLLAMA_BASE_URL)

    def generate(self,prompt: str) -> str:
        return self.llm.invoke(prompt)
    
    def generate_with_context(
        self,
        query: str,
        documents: list[str],
        instruction: str | None = None
    ) -> str:
        context = "\n".join(documents)

        system_prompt = instruction or "Answer the question strictly using the provided context. "
        "If the answer is not present, say you don't know."

        prompt = f"""
    {system_prompt}

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else response