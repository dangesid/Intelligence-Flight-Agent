from typing import Dict, Any
from src.agents.base_agent import BaseAgent
import re


class QueryIntentAgent(BaseAgent):
    """
    Determines user intent from the query.
    Runs exactly once.
    """

    def can_handle(self, payload: Dict[str, Any]) -> bool:
        return "query" in payload and "intent" not in payload

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        query = payload["query"].lower()

        # Lightweight intent inference (not routing)
        flight_id_match = re.search(r"\bflight\s*(\d+)\b", query)

        if flight_id_match:
            payload["intent"] = {
                "type": "flight_id_lookup",
                "flight_number": flight_id_match.group(1)
            }
        else:
            payload["intent"] = {
                "type": "general_flight_query"
            }

        return payload
