import re

class QueryIntentAgents:
    def classify(self, query: str) -> dict:
        q = query.lower()

        match = re.search(r"\bflight\s+(\d+)\b", q)
        if match:
            return {
                "intent": "flight_id_lookup",
                "flight_number": match.group(1)
            }

        if "from" in q and "to" in q:
            return {"intent": "route_query"}

        if "delay" in q or "status" in q:
            return {"intent": "status_query"}

        # 🆕 SYSTEM / DATASET CAPABILITY
        if any(k in q for k in [
            "what routes",
            "what data",
            "what flights do you have",
            "coverage",
            "available routes",
            "do you have data"
        ]):
            return {"intent": "system_capability"}

        return {"intent": "general"}
