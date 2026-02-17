# src/agents/context_builder_agent.py
from typing import Dict, Any, List, Optional
from src.agents.base_agent import BaseAgent


class ContextBuilderAgent(BaseAgent):
    """
    Agentic Context Builder
    - Converts structured retrieved_docs into LLM-readable text
    - Evaluates autonomously whether it should act
    - Can propose next steps/tools
    """

    def __init__(self):
        super().__init__(name="ContextBuilderAgent")

    # -------------------
    # LEGACY METHODS
    # -------------------
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
        return self.execute(payload)

    # -------------------
    # AGENTIC METHODS
    # -------------------
    def evaluate(self, payload: Dict[str, Any]) -> bool:
        """
        Decide if ContextBuilderAgent should act
        """
        return self.can_handle(payload)

    def propose_action(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Suggest ClaraAgent as the next step after context is built
        """
        return {
            "next_agents": ["ClaraAgent"],
            "tools_to_call": [],
            "modify_payload": {}
        }

    def execute(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        docs = payload.get("retrieved_docs", [])

        if not docs:
            payload["context"] = None  # do not block Clara
            return payload

        context_chunks: List[str] = []

        for i, row in enumerate(docs[:10], start=1):
            if isinstance(row, str):
                context_chunks.append(f"[Doc {i}] {row}")
            elif isinstance(row, dict):
                readable_fields = [f"{k.replace('_',' ').capitalize()}: {v}" for k, v in row.items()]
                doc_text = "; ".join(readable_fields)
                context_chunks.append(f"[Doc {i}] {doc_text}")
            else:
                context_chunks.append(f"[Doc {i}] {str(row)}")

        context = "\n".join(context_chunks).strip()
        payload["context"] = context if context else None

        print("\n📝 Debug - Context built from retrieved docs:")
        print(payload["context"])
        return payload
