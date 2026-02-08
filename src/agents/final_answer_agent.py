from typing import Dict, Any
from src.agents.base_agent import BaseAgent


class FinalAnswerAgent(BaseAgent):
    """
    Declares completion once an answer exists.
    """

    def can_handle(self, payload: Dict[str, Any]) -> bool:
        return "answer" in payload and not payload.get("finalized")

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        payload["finalized"] = True
        return payload
