from typing import Dict, Any
from statistics import mean
from src.agents.base_agent import BaseAgent


class CoverageEvaluatorAgent(BaseAgent):
    """
    Tracks KB coverage trends over time.

    - No judgments
    - No thresholds
    - Signal accumulation only
    """

    def can_handle(self, payload: Dict[str, Any]) -> bool:
        return (
            "coverage_history" in payload
            and len(payload["coverage_history"]) >= 3
            and "coverage_trends" not in payload
        )

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        history = payload["coverage_history"]

        payload["coverage_trends"] = {
            "average_coverage": mean(history),
            "latest_coverage": history[-1],
            "delta": history[-1] - history[0],
            "samples": len(history)
        }

        return payload


"""
The Payload will look like this:

"coverage_trends": {
  "average_coverage": 0.61,
  "latest_coverage": 0.72,
  "delta": 0.18,
  "samples": 14
}


"""