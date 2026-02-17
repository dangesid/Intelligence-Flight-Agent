# src/agents/coverage_evaluator_agent.py
from typing import Dict, Any, Optional
from statistics import mean
from src.agents.base_agent import BaseAgent


class CoverageEvaluatorAgent(BaseAgent):
    """
    Agentic Coverage Evaluator
    - Tracks KB coverage trends over time.
    - Does not modify system behavior, just updates coverage stats.
    """

    def __init__(self):
        super().__init__(name="CoverageEvaluatorAgent")

    # -------------------
    # LEGACY METHODS
    # -------------------
    def can_handle(self, payload: Dict[str, Any]) -> bool:
        return (
            "coverage_history" in payload
            and len(payload["coverage_history"]) >= 3
            and "coverage_trends" not in payload
        )

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.execute(payload)

    # -------------------
    # AGENTIC METHODS
    # -------------------
    def evaluate(self, payload: Dict[str, Any]) -> bool:
        """
        Decide if CoverageEvaluatorAgent should act
        """
        return self.can_handle(payload)

    def propose_action(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Suggest next agent/tool if needed
        Currently, no next agents or tools; structure is ready for agentic orchestration
        """
        return None  # This agent just updates stats, does not route

    def execute(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        history = payload.get("coverage_history", [])

        if not history or len(history) < 3 or "coverage_trends" in payload:
            return payload  # Nothing to do

        payload["coverage_trends"] = {
            "average_coverage": mean(history),
            "latest_coverage": history[-1],
            "delta": history[-1] - history[0],
            "samples": len(history)
        }

        return payload
