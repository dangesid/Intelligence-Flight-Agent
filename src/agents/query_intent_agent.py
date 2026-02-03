class QueryIntentAgents:
    def classify(self, query: str)-> dict:
        q = query.lower()
        if "flight" in q and any(x.isdigit() for x in q):
            return {"intent": "flight_specific"}

        if "from" in q and "to" in q:
            return {"intent": "route_query"}

        if "delay" in q or "status" in q:
            return {"intent": "status_query"}

        return {"intent": "general"}