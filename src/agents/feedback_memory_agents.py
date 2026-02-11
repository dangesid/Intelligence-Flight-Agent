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
        # Only store gaps if gap_signal exists and feedback hasn't been captured
        return "query" in payload and "gap_signal" in payload and "feedback_memory" not in payload

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        gap_signal = payload.get("gap_signal")
        if not gap_signal:
            return payload

        # Generate LLM analysis of the gap
        analysis = self._analyze_knowledge_gap(payload)

        # Append gap to payload memory
        payload.setdefault("knowledge_gaps", [])
        payload["knowledge_gaps"].append({
            "question": payload.get("query"),
            "system_answer": payload.get("answer", {}).get("text"),
            "gap_reason": gap_signal.get("reason"),
            "severity": gap_signal.get("severity"),
            "missing_knowledge": payload.get("missing_knowledge"),
            "llm_analysis": analysis
        })

        # Persist to system_state.json
        self._persist_to_system_state(payload["knowledge_gaps"])

        # Mark feedback captured
        payload["feedback_memory"] = True

        print("✅ Knowledge gap logged successfully.")
        return payload

    # ==========================================================
    # Internal helpers
    # ==========================================================

    def _analyze_knowledge_gap(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to suggest what knowledge is missing."""
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

Task:
Respond ONLY in JSON with the following structure:

{{
  "knowledge_gap": "<what is missing>",
  "recommended_data_to_add": [
    "<data type 1>",
    "<data type 2>"
  ]
}}
"""
        try:
            return self.llm.generate_json(prompt)
        except Exception:
            # fallback if LLM fails
            return {
                "knowledge_gap": gap_signal.get("reason", "Unknown"),
                "recommended_data_to_add": []
            }

    def _persist_to_system_state(self, knowledge_gaps: list):
        """Append gaps to persistent system state."""
        state = {"knowledge_gaps": []}
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r") as f:
                    state = json.load(f)
            except Exception:
                state = {"knowledge_gaps": []}

        state.setdefault("knowledge_gaps", [])
        state["knowledge_gaps"].extend(knowledge_gaps)

        with open(self.state_path, "w") as f:
            json.dump(state, f, indent=2)
