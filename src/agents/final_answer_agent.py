# src/agents/final_answer_agent.py

from typing import Dict, Any, Optional
from src.agents.base_agent import BaseAgent


class FinalizerAgent(BaseAgent):
    """
    Agentic Finalizer Agent
    - Ensures system always terminates cleanly
    - Can evaluate autonomously and propose next steps (if needed)
    """

    def __init__(self):
        super().__init__(name="FinalizerAgent")

    # -------------------
    # LEGACY METHODS
    # -------------------
    def can_handle(self, payload: Dict[str, Any]) -> bool:
        # Run once after answer exists
        return (
            payload.get("answer") is not None
            and not payload.get("finalized", False)
        )

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.execute(payload)

    # -------------------
    # AGENTIC METHODS
    # -------------------
    def evaluate(self, payload: Dict[str, Any]) -> bool:
        """
        Decide if FinalizerAgent should act
        """
        return self.can_handle(payload)

    def propose_action(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        FinalizerAgent does not call other agents/tools
        """
        return None

    def execute(self, payload: Dict[str, Any]) -> Dict[str, Any]:
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
