from typing import Dict, Any, List
from src.agents.base_agent import BaseAgent


class KnowledgeGapAgent(BaseAgent):
    """
    Autonomous Knowledge Gap Detection Agent

    Responsibilities:
    - Inspect retrieval results
    - Decide if information is insufficient, ambiguous, or out-of-domain
    - Produce a structured gap_report
    - Never block other agents explicitly
    """

    def can_handle(self, payload: Dict[str, Any]) -> bool:
        """
        Run if:
        - A query exists
        - Retrieval has already happened
        - No final answer has been produced yet
        """
        return (
            "query" in payload
            and "retrieved_docs" in payload
            and "answer" not in payload
            and "gap_report" not in payload
        )

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        query: str = payload.get("query", "")
        docs: List[str] = payload.get("retrieved_docs", [])
        intent = payload.get("intent", {})

        # -------------------------
        # Case 1: No data at all
        # -------------------------
        if not docs:
            payload["gap_report"] = {
                "gap_type": "NO_DATA",
                "reason": "No relevant documents were found for this query.",
                "query": query
            }
            return payload

        # -------------------------
        # Case 2: Too generic / ambiguous
        # -------------------------
        unique_signals = set()
        for doc in docs:
            parts = doc.split(",")
            for p in parts:
                if "from" in p.lower() or "to" in p.lower():
                    unique_signals.add(p.strip())

        if len(unique_signals) > 5:
            payload["gap_report"] = {
                "gap_type": "AMBIGUOUS_MATCH",
                "reason": "Multiple possible matches found. Query is underspecified.",
                "query": query
            }
            return payload

        # -------------------------
        # Case 3: Capability gap
        # -------------------------
        unsupported_keywords = ["price", "delay", "status", "cancel", "weather"]
        if any(k in query.lower() for k in unsupported_keywords):
            payload["gap_report"] = {
                "gap_type": "CAPABILITY_GAP",
                "reason": "The system does not support this type of information.",
                "query": query
            }
            return payload

        # -------------------------
        # Otherwise: No gap detected
        # -------------------------
        return payload
