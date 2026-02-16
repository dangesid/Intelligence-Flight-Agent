# src/agents/final_answer_agent.py

from typing import Dict, Any
from src.agents.base_agent import BaseAgent


class FinalizerAgent(BaseAgent):
    """
    Final step of orchestration.
    Ensures system always terminates cleanly.
    """

    def can_handle(self, payload: Dict[str, Any]) -> bool:
        # Run once after answer exists
        return (
            payload.get("answer") is not None
            and not payload.get("finalized", False)
        )

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:

        gap_signal = payload.get("gap_signal")

        # 🚨 HARD GATE: High severity gap blocks answer
        if gap_signal and gap_signal.get("severity") == "high":
            payload["answer"] = {
                "text": "Data not available ⚠ Reason: "
                        + gap_signal.get("reason", "High severity knowledge gap."),
                "confidence": 0.3
            }

        # 🔥 CRITICAL: Mark orchestration complete
        payload["finalized"] = True

        return payload
