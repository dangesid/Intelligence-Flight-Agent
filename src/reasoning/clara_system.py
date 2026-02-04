from typing import List, Dict, Optional, Any
from src.llm_engine.factory import get_llm


class ClaraSystem:
    """
    CLARA = Controlled, Limited, Anchored Response Agent

    Responsibilities:
    - Grounded answers from retrieved docs
    - Deterministic summaries for broad questions
    - Explicit gap explanations
    - Never hallucinate
    """

    def __init__(self, llm: Optional[Any] = None):
        self.llm = llm or get_llm()

    # ==================================================
    # 🔍 BROAD / SUMMARY QUESTION DETECTOR
    # ==================================================
    def _is_summary_query(self, query: str) -> bool:
        q = query.lower()
        return any(p in q for p in [
            "what flight information",
            "what flights",
            "flight information",
            "give me information",
            "information about flights",
            "what do you have",
            "flight data"
        ])

    # ==================================================
    # 📊 SAFE DOC-BASED SUMMARY (NO LLM)
    # ==================================================
    def _summarize_docs(self, docs: List[str]) -> str:
        flights = set()
        origins = set()
        destinations = set()
        carriers = set()

        for doc in docs:
            parts = doc.split(",")

            for p in parts:
                p = p.strip()
                if p.lower().startswith("flight"):
                    flights.add(p)
                elif p.lower().startswith("from"):
                    origins.add(p.replace("from", "").strip())
                elif p.lower().startswith("to"):
                    destinations.add(p.replace("to", "").strip())
                elif "carrier" in p.lower():
                    carriers.add(p.split()[-1])

        return (
            "I have scheduled flight information including:\n\n"
            f"• Example flights: {', '.join(list(flights)[:5]) or 'N/A'}\n"
            f"• Origins: {', '.join(list(origins)[:5]) or 'N/A'}\n"
            f"• Destinations: {', '.join(list(destinations)[:5]) or 'N/A'}\n"
            f"• Carriers: {', '.join(list(carriers)[:5]) or 'N/A'}\n\n"
            "Ask about a specific flight number or route for details."
        )

    # ==================================================
    # ✅ GROUNDED ANSWER (DETAIL MODE)
    # ==================================================
    def answer(
        self,
        query: str,
        docs: List[str],
        confidence: float,
        cautious: bool = False
    ) -> Dict[str, Any]:

        if not docs:
            return {
                "answer": "I don’t have enough relevant flight data to answer this question.",
                "confidence": 0.0
            }

        # 🟢 SUMMARY MODE (OVERRIDES LLM)
        if self._is_summary_query(query):
            return {
                "answer": self._summarize_docs(docs),
                "confidence": round(confidence, 2)
            }

        # 🔵 DETAIL MODE (LLM, BUT GROUNDED)
        instruction = (
            "Answer strictly using the provided flight data.\n"
            "- Do NOT guess or assume missing facts\n"
            "- Do NOT use external knowledge\n"
            "- If the answer is not present, say so explicitly"
        )

        if cautious:
            instruction += (
                "\nThe match is partial, so be conservative and explicit."
            )

        try:
            answer_text = self.llm.generate_with_context(
                query=query,
                context_docs=docs,
                instruction=instruction
            )
        except Exception:
            return {
                "answer": (
                    "I found related flight data, but cannot confidently "
                    "answer this question based on the available information."
                ),
                "confidence": round(confidence * 0.6, 2)
            }

        return {
            "answer": str(answer_text),
            "confidence": round(confidence, 2)
        }

    # ==================================================
    # ❌ NO-ANSWER PATH (SAFE EXIT)
    # ==================================================
    def no_answer(
        self,
        gap_report: Dict[str, Any],
        confidence: float
    ) -> Dict[str, Any]:

        if not isinstance(gap_report, dict):
            return {
                "answer": "I’m unable to answer this question with the available data.",
                "confidence": 0.0
            }

        return {
            "answer": self.respond(gap_report),
            "confidence": round(confidence, 2)
        }

    # ==================================================
    # 📦 DATASET INVENTORY ANSWER (NO RETRIEVAL)
    # ==================================================
    def dataset_answer(self, domain_summary: Dict[str, Any]) -> Dict[str, Any]:

        return {
            "answer": (
                "Here’s what I currently have data for:\n\n"
                f"• Flight numbers: {', '.join(map(str, domain_summary.get('flights', [])[:10]))}\n"
                f"• Origins: {', '.join(domain_summary.get('origins', [])[:10])}\n"
                f"• Destinations: {', '.join(domain_summary.get('destinations', [])[:10])}\n"
                f"• Carriers: {', '.join(domain_summary.get('carriers', [])[:10])}\n\n"
                "Ask about a specific flight or route for more details."
            ),
            "confidence": 0.95
        }

    # ==================================================
    # 🧠 HUMAN-FRIENDLY GAP EXPLANATION
    # ==================================================
    def respond(self, gap_report: Dict[str, Any]) -> str:

        gap_type = gap_report.get("gap_type", "UNKNOWN")

        if gap_type == "DOMAIN_GAP":
            domains = gap_report.get("available_domains", {})
            return (
                "I don’t have data for that request.\n\n"
                f"Available origins: {', '.join(domains.get('origins', [])[:5])}\n"
                f"Available destinations: {', '.join(domains.get('destinations', [])[:5])}\n\n"
                "Try asking about a specific route or flight number."
            )

        if gap_type == "WEAK_MATCH":
            return (
                "I found related flight data, but your question is incomplete.\n"
                "Please specify a flight number, route, or airline."
            )

        if gap_type == "AMBIGUOUS_MATCH":
            return (
                "Multiple flights match your request.\n"
                "Please clarify with origin, destination, or carrier."
            )

        if gap_type == "CAPABILITY_GAP":
            return (
                "This system only supports scheduled flight information.\n"
                "It does not provide pricing, delays, or real-time status."
            )

        return "I’m unable to confidently answer this question with the available data."