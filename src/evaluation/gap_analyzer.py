"""
This step will answer questions like:
Why did the system reject my question?
Is the data missing?
Is the query out of domain?
Was something similar retrieved but filtered out?
"""

from typing import List, Dict

class KnowledgeGapAnalyzer:
    """
    Explains WHY the system could not confidently answer a query.
    """

    def analyze(
        self,
        query: str,
        intent: Dict,
        distance: List[float],
        documents: List[str],
        metadatas: List[Dict],
        domain_summary: Dict
    ) -> str:

        best_distance = min(distance)

        # 1️⃣ Capability gap
        if not intent.get("supported", True):
            if intent.get("requires_realtime", False):
                return (
                    "This question requires real-time flight information such as delays or live status. "
                    "The current system only supports static flight schedule data."
                )

            return (
                "This type of question is not supported by the current system."
            )

        # 2️⃣ Intent-level ambiguity
        if intent.get("ambiguous", False):
            return (
                "Your question is ambiguous. Please provide more details such as "
                "origin, destination, flight number, or airline."
            )

        # 3️⃣ Out-of-domain
        if best_distance > 0.95:
            origins = domain_summary.get("origins", [])[:5]
            destinations = domain_summary.get("destinations", [])[:5]

            route_text = "\n".join(
                f"- {o} → {d}" for o, d in zip(origins, destinations)
            )

            return (
                "Your question is outside the scope of the available data.\n\n"
                "This system currently contains flight data for routes such as:\n"
                f"{route_text}\n\n"
                "Try asking about one of these routes or a specific flight number."
            )

        # 4️⃣ Weak semantic match
        if best_distance > 0.85:
            return (
                "Some related flight information exists, but the question is incomplete.\n\n"
                "Try specifying:\n"
                "- Origin and destination\n"
                "- Flight number\n"
                "- Airline"
            )

        # 5️⃣ Entity ambiguity
        flight_ids = {m.get("flight") for m in metadatas if m.get("flight")}
        if len(flight_ids) > 1:
            return (
                "Multiple similar flights were found. "
                "Please specify origin, destination, or airline."
            )

        # Fallback
        return (
            "The system could not confidently determine a single relevant answer "
            "from the available data."
        )
