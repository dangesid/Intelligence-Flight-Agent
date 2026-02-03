from abc import ABC, abstractmethod
from typing import List

class BaseLLM(ABC):
    """
    Docstring for BaseLLM
    """

    @abstractmethod
    def generate(self,prompt: str) -> str:
        pass
    @abstractmethod
    def generate_with_context(self, prompt: str, context: List[str]) -> str:
        pass 
