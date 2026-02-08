from typing import Dict, Any, List
from src.agents.base_agent import BaseAgent


class ContextBuilderAgent(BaseAgent):
    """
    Converts structured retrieved_docs into LLM-readable text
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

            # Case 1: already text (most likely your case)
            if isinstance(row, str):
                context_chunks.append(f"[Doc {i}] {row}")

            # Case 2: structured dict (future-proof)
            elif isinstance(row, dict):
                context_chunks.append(
                    f"Flight {row.get('flight')} from {row.get('origin')} to {row.get('dest')}, "
                    f"carrier {row.get('carrier')}, "
                    f"departs at {row.get('dep_time')}, arrives at {row.get('arr_time')}, "
                    f"distance {row.get('distance')} miles."
                )

            # Fallback
            else:
                context_chunks.append(f"[Doc {i}] {str(row)}")

        context = "\n".join(context_chunks).strip()

        if not context:
            payload["context"] = "NO_RELEVANT_DATA"
        else:
            payload["context"] = context

        return payload
