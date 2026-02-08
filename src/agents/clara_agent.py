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
    """

    def __init__(self, llm=None):
        self.llm = llm or get_llm()

    def can_handle(self, payload: Dict[str, Any]) -> bool:
        # Clara runs ONLY if:
        # - Query exists
        # - System not finalized
        # - Either docs OR gap report exists
        # - No answer yet
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
        context = "\n".join(docs[:5]) if docs else "NO_DATA"

        prompt = f"""
You are CLARA.
Answer STRICTLY using the provided data.

Question:
{query}

Data:
{context}

Rules:
- If answer exists, answer clearly
- If data is insufficient, say so
- NO hallucination
- NO external knowledge
"""

        try:
            response = self.llm.generate(prompt)
        except Exception:
            payload["answer"] = {
                "text": "Unable to generate an answer with the current data.",
                "confidence": 0.0
            }
            payload["finalized"] = True
            return payload

        payload["answer"] = {
            "text": response.strip(),
            "confidence": 0.7 if docs else 0.0
        }

        # ✅ TERMINATE SYSTEM
        payload["finalized"] = True
        return payload
