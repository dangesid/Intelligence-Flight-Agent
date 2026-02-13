# src/agents/context_builder_agent.py
from typing import Dict, Any, List
from src.agents.base_agent import BaseAgent

class ContextBuilderAgent(BaseAgent):
    """
    Converts structured retrieved_docs into LLM-readable text
    """

    def can_handle(self, payload: Dict[str, Any]) -> bool:
        answer = payload.get("answer", {})
        context = payload.get("context")
        return "query" in payload and not context and payload.get("retrieved_docs") and not answer.get("text")

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        docs = payload.get("retrieved_docs", [])

        if not docs:
            payload["context"] = None  # do not block Clara
            return payload

        context_chunks: List[str] = []

        for i, row in enumerate(docs[:10], start=1):
            if isinstance(row, str):
                context_chunks.append(f"[Doc {i}] {row}")
            elif isinstance(row, dict):
                readable_fields = [f"{k.replace('_',' ').capitalize()}: {v}" for k,v in row.items()]
                doc_text = "; ".join(readable_fields)
                context_chunks.append(f"[Doc {i}] {doc_text}")
            else:
                context_chunks.append(f"[Doc {i}] {str(row)}")

        context = "\n".join(context_chunks).strip()
        payload["context"] = context if context else None

        print("\n📝 Debug - Context built from retrieved docs:")
        print(payload["context"])
        return payload
