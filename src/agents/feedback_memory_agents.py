# src/agents/feedback_memory_agents.py
import os
import json
import re
from typing import Dict, Any
from src.agents.base_agent import BaseAgent
from src.llm_clients.azure_openai_client import AzureOpenAIWrapper
from langchain_core.messages import HumanMessage


class FeedbackMemoryAgent(BaseAgent):
    """
    Stores knowledge gaps into persistent system_state.json safely.
    """

    def __init__(self, state_path="system_state.json"):
        # Create its own LLM instance
        self.llm = AzureOpenAIWrapper()
        self.state_path = state_path

    def can_handle(self, payload: dict) -> bool:
        return "gap_signal" in payload and payload["gap_signal"] and not payload.get("_feedback_done", False)

    def run(self, payload: dict) -> dict:
        gap_signal = payload.get("gap_signal")
        if not gap_signal:
            return payload

        print("\n🧠 [FeedbackMemoryAgent] Logging knowledge gap...")
        self._log_knowledge_gap(payload)

        # Mark as done
        payload["_feedback_done"] = True
        return payload

    # ==========================================================
    # Internal helpers
    # ==========================================================

    def _load_system_state(self) -> Dict[str, Any]:
        if not os.path.exists(self.state_path):
            return {"knowledge_gaps": []}
        try:
            state = json.loads(open(self.state_path).read())
            if isinstance(state, list):  # old format
                state = {"knowledge_gaps": state}
            if "knowledge_gaps" not in state:
                state["knowledge_gaps"] = []
            return state
        except Exception:
            return {"knowledge_gaps": []}

    def _save_system_state(self, state: Dict[str, Any]):
        with open(self.state_path, "w") as f:
            json.dump(state, f, indent=2)

    def _persist_to_system_state(self, knowledge_gaps: list):
        state = self._load_system_state()
        if not isinstance(state, dict):
            state = {"knowledge_gaps": []}  # fix for list fallback

        state.setdefault("knowledge_gaps", [])
        state["knowledge_gaps"].extend(knowledge_gaps)
        self._save_system_state(state)

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
            # Use the wrapper's generate method
            response = self.llm.generate(messages=[HumanMessage(content=prompt)])
            
            # Extract JSON from LLM text safely
            json_text = response["choices"][0]["message"]["content"]
            match = re.search(r"\{.*\}", json_text, flags=re.DOTALL)
            if match:
                try:
                    analysis_json = json.loads(match.group())
                except json.JSONDecodeError:
                    analysis_json = {
                        "knowledge_gap": gap_signal.get("reason", "Unknown"),
                        "recommended_data_to_add": []
                    }
            else:
                analysis_json = {
                    "knowledge_gap": gap_signal.get("reason", "Unknown"),
                    "recommended_data_to_add": []
                }
            
        except Exception as e:
            print(f"⚠️ FeedbackMemoryAgent LLM Error: {e}")
            import traceback
            traceback.print_exc()
            
            analysis_json = {
                "knowledge_gap": gap_signal.get("reason", "Unknown"),
                "recommended_data_to_add": []
            }

        payload.setdefault("knowledge_gaps", [])
        payload["knowledge_gaps"].append({
            "question": query,
            "system_answer": answer,
            "gap_reason": gap_signal.get("reason"),
            "severity": gap_signal.get("severity"),
            "llm_analysis": analysis_json
        })
        print("✅ Knowledge gap added to payload memory")
