from typing import Dict, Any, List
from src.llm_engine.factory import get_llm
from src.agents.base_agent import BaseAgent


class ClaraAgent(BaseAgent):
    """
    CLARA – Controlled, Limited, Anchored Response Agent

    FINAL ANSWER AGENT
    - Runs once
    - Produces answer OR refusal
    - Terminates orchestration
    - Fully grounded on provided context (retrieved_docs or gap_report)
    """

    def __init__(self, llm=None):
        # Initialize LLM; can pass custom LLM instance
        self.llm = llm or get_llm()

    def can_handle(self, payload: Dict[str, Any]) -> bool:
        """
        Clara runs ONLY if:
        - Query exists
        - System not finalized
        - Either retrieved_docs OR gap_report exists
        - No answer yet
        """
        return (
            "query" in payload
            and not payload.get("finalized")
            and "answer" not in payload
            and (
                payload.get("retrieved_docs") is not None
                or payload.get("gap_report") is not None
            )
        )

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        query: str = payload["query"]
        docs: List[str] = payload.get("retrieved_docs", [])
        gap_report = payload.get("gap_report")

        # 🚫 GAP RESPONSE (NO LLM)
        if gap_report:
            payload["answer"] = {
                "text": gap_report.get(
                    "message",
                    "I cannot answer this question with the available data."
                ),
                "confidence": 0.2
            }
            payload["finalized"] = True
            return payload

        # 📚 DOC-BASED ANSWER
        if docs:
            try:
                response = self.llm.generate_with_context(
                    query=query,
                    documents=docs,
                    instruction=(
                        "Answer strictly using the provided data. "
                        "If the information is not present, say 'Data not available'. "
                        "Do NOT use any external knowledge or make assumptions."
                    )
                )
            except Exception:
                response = "Unable to generate an answer with the current data."
        else:
            response = "Data not available."

        payload["answer"] = {
            "text": response.strip(),
            "confidence": 0.7 if docs else 0.0
        }

        # ✅ TERMINATE SYSTEM
        payload["finalized"] = True
        return payload
