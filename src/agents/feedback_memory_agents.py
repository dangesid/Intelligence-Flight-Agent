import json
import os
from typing import Dict, Any
from src.agents.base_agent import BaseAgent


class FeedbackMemoryAgent(BaseAgent):
    """
    Stores knowledge gaps into persistent system_state.json.

    - Compatible with new gap_signal format
    - Robust to missing fields
    - Uses LLM to analyze missing knowledge
    """

    def __init__(self, llm_client, state_path="system_state.json"):
        self.llm = llm_client
        self.state_path = state_path

    def can_handle(self, payload: Dict[str, Any]) -> bool:
        return "query" in payload and "gap_signal" in payload

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        gap_signal = payload.get("gap_signal")

        if not gap_signal:
            return payload

        print("\n🧠 [FeedbackMemoryAgent] Logging knowledge gap...")

        self._log_knowledge_gap(payload)

        payload["feedback_memory"] = True
        return payload

    # ==========================================================
    # Internal helpers
    # ==========================================================

    def _load_system_state(self) -> Dict[str, Any]:
        if not os.path.exists(self.state_path):
            return {"knowledge_gaps": []}

        with open(self.state_path, "r") as f:
            return json.load(f)

    def _save_system_state(self, state: Dict[str, Any]):
        with open(self.state_path, "w") as f:
            json.dump(state, f, indent=2)

    def _log_knowledge_gap(self, payload: Dict[str, Any]):
        query = payload.get("query")
        answer = payload.get("answer", {}).get("text")
        gap_signal = payload.get("gap_signal", {})

        prompt = f"""
    You are an AI system analyst helping improve a knowledge base.

    User question:
    {query}

    System answer:
    {answer}

    Detected gap signal:
    {gap_signal}

    Respond ONLY in JSON:

    {{
    "knowledge_gap": "<what is missing>",
    "recommended_data_to_add": [
        "<data type 1>",
        "<data type 2>"
    ]
    }}
    """

        try:
            analysis = self.llm.generate_json(prompt)
        except Exception:
            analysis = {
                "knowledge_gap": gap_signal.get("reason", "Unknown"),
                "recommended_data_to_add": []
            }

        payload.setdefault("knowledge_gaps", [])

        payload["knowledge_gaps"].append({
            "question": query,
            "system_answer": answer,
            "gap_reason": gap_signal.get("reason"),
            "severity": gap_signal.get("severity"),
            "llm_analysis": analysis
        })

        print("✅ Knowledge gap added to payload memory")
