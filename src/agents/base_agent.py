from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseAgent(ABC):
    """
    Docstring for BaseAgent
    """

    @abstractmethod
    def can_handle(self, payload: Dict[str, Any]) -> bool:
        """
        Return True if this agent should act on the given payload.
        """
        pass
    
    @abstractmethod
    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the given payload and return a result updates.
        """
        pass