from typing import Dict, Any
from src.agents.base_agent import BaseAgent


class FinalizerAgent(BaseAgent):
    """
    Ends orchestration only after learning agents ran.
    """

    def can_handle(self, payload: Dict[str, Any]) -> bool:
        return (
            payload.get("answer_ready") is True
            and "finalized" not in payload
        )

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        gap_signal = payload.get("gap_signal", {})

        # 🚨 HARD GATE: High severity gap = no answer generation
        if gap_signal.get("severity") == "high":
            payload["answer"] = {
                "text": "I could not find relevant flight information for your requested destination in the knowledge base.",
                "confidence": 0.3
            }
            return payload
