class KnowledgeGapAgent:
    """
    Explains WHY CLARA cannot answer
    """

    def analyze(
        self,
        query: str,
        intent: str,
        distances: list,
        metadatas: list,
        domain_summary: dict
    ) -> str:

        best_distance = min(distances)

        # -----------------------------
        # System capability gaps
        # -----------------------------
        if intent == "system_capability":
            return (
                "This question is about system capabilities.\n"
                "I can answer questions about scheduled flights, routes, carriers, "
                "and flight numbers present in my dataset."
            )

        # -----------------------------
        # Weak semantic match
        # -----------------------------
        if best_distance > 0.85 and best_distance <= 0.95:
            return (
                "I found flight data that is somewhat related, "
                "but not enough to confidently answer your question.\n\n"
                "Try being more specific, for example:\n"
                "• Flights from LGA to ATL\n"
                "• Details of Flight 347"
            )

        # -----------------------------
        # Domain gap
        # -----------------------------
        origins = domain_summary.get("origins", [])
        destinations = domain_summary.get("destinations", [])

        return (
            "I don’t have enough information to answer that question.\n\n"
            "My data currently covers routes such as:\n"
            f"- Origins: {', '.join(origins[:5])}\n"
            f"- Destinations: {', '.join(destinations[:5])}\n"
        )
