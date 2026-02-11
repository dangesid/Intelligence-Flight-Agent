from typing import Dict, Any
from src.agents.base_agent import BaseAgent


class KnowledgeGapAgent(BaseAgent):
    """
    Detects knowledge gaps by validating whether retrieved context
    is actually relevant to the user's query.
    """

    def __init__(self, llm_client):
        super().__init__()
        self.llm = llm_client

    def can_handle(self, payload: Dict[str, Any]) -> bool:
        # Only run if we have a query and retrieved context
        return "query" in payload and "context" in payload

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        query = payload.get("query", "")
        context = payload.get("context", "")

        # 🚨 Case 1: Nothing retrieved
        if not context:
            payload["gap_signal"] = {
                "reason": "No documents retrieved from knowledge base.",
                "severity": "high"
            }
            payload["missing_knowledge"] = "Relevant domain knowledge not present in vector store."
            return payload

        prompt = f"""
    You are a strict JSON validation engine.

    User Query:
    {query}

    Retrieved Context:
    {context}

    Return STRICT JSON only. No explanation. No markdown.

    {{
    "relevant": true or false,
    "answerable": true or false,
    "reason": "short explanation",
    "severity": "low" | "medium" | "high",
    "missing_knowledge": "what is missing"
    }}
    """

        raw_response = self.llm.generate(prompt)

        import json
        import re

        try:
            match = re.search(r"\{.*", raw_response, re.DOTALL)
            if not match:
                raise ValueError("No JSON found")

            json_str = match.group()

            # 🔥 Auto-fix missing closing brace
            if not json_str.strip().endswith("}"):
                json_str = json_str.strip() + "}"

            result = json.loads(json_str)
        except Exception:
            print("⚠️ LLM RAW OUTPUT (Parsing Failed):")
            print(raw_response)

            # ⚠️ IMPORTANT: Do NOT treat parse failure as real gap
            return payload  # Just continue without creating gap_signal

        # Only create gap if actually not relevant or not answerable
        if not result.get("relevant", True) or not result.get("answerable", True):
            payload["gap_signal"] = {
                "reason": result.get("reason"),
                "severity": result.get("severity", "medium")
            }
            payload["missing_knowledge"] = result.get("missing_knowledge")

        return payload
