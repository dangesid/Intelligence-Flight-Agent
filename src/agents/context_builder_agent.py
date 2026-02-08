from typing import Dict, Any, List
from src.agents.base_agent import BaseAgent

class ContextBuilderAgent(BaseAgent):
    """
    Converts structured retrieved_docs into LLM-readable text
    Works for any kind of data, not specific to flights
    """

    def can_handle(self, payload: Dict[str, Any]) -> bool:
        answer = payload.get("answer", {})
        context = payload.get("context")
        return (
            "query" in payload
            and not context
            and payload.get("retrieved_docs")
            and not answer.get("text")
        )

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        docs = payload.get("retrieved_docs", [])

        if not docs:
            payload["context"] = "NO_RELEVANT_DATA"
            return payload

        context_chunks: List[str] = []

        for i, row in enumerate(docs[:10], start=1):

            # Case 1: already text
            if isinstance(row, str):
                context_chunks.append(f"[Doc {i}] {row}")

            # Case 2: structured dict
            elif isinstance(row, dict):
                # Convert all key-value pairs into readable text
                fields = [f"{k}: {v}" for k, v in row.items()]
                doc_text = ", ".join(fields)
                context_chunks.append(f"[Doc {i}] {doc_text}")

            # Fallback
            else:
                context_chunks.append(f"[Doc {i}] {str(row)}")

        context = "\n".join(context_chunks).strip()

        payload["context"] = context if context else "NO_RELEVANT_DATA"

        return payload
