from typing import Dict, Any, List
from src.llm_engine.factory import get_llm
from src.agents.base_agent import BaseAgent


class ClaraAgent(BaseAgent):
    """
    CLARA – Controlled, Limited, Anchored Response Agent

    - Fully agentic (no if/else routing)
    - Reads shared payload
    - Answers strictly from retrieved_docs
    - Refuses gracefully when data is insufficient
    """

    def __init__(self, llm=None):
        super().__init__()
        self.llm = llm or get_llm()

    def can_handle(self, payload: Dict[str, Any]) -> bool:
        answer = payload.get("answer")

        retrieved_docs = payload.get("retrieved_docs") or []
        gap_report = payload.get("gap_report")

        return (
            isinstance(payload.get("query"), str)
            and (retrieved_docs or gap_report)
            and (not answer or not answer.get("text"))
        )

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        query: str = payload["query"]
        docs: List[str] = payload.get("retrieved_docs", [])
        gap_report = payload.get("gap_report")

        context_text = (
            "\n".join(docs[:5])
            if docs
            else "NO_RELEVANT_DATA"
        )

        gap_text = (
            f"\nKnowledge Gaps Identified:\n{gap_report}"
            if gap_report
            else ""
        )

        prompt = f"""
You are CLARA, an agent that answers questions ONLY using provided data.

User Question:
{query}

Available Data:
{context_text}
{gap_text}

Rules:
- Answer ONLY if the data explicitly supports it
- If data is missing or insufficient, say so clearly
- Do NOT hallucinate
- Do NOT add external knowledge

Return strictly in this format:

Answer:
<answer text>

Confidence:
<number between 0.0 and 1.0>
"""

        try:
            response = self.llm.generate(prompt)
        except Exception as e:
            payload["answer"] = {
                "text": "Answer generation failed due to an internal error.",
                "confidence": 0.0,
                "error": str(e),
            }
            return payload

        # --- Parse response safely ---
        answer_text = response
        confidence = 0.0

        if "Confidence:" in response:
            parts = response.split("Confidence:")
            answer_text = parts[0].replace("Answer:", "").strip()
            try:
                confidence = float(parts[1].strip())
            except ValueError:
                confidence = 0.2

        payload["answer"] = {
            "text": answer_text,
            "confidence": confidence,
        }

        return payload
