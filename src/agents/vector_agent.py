from typing import Dict, Any
from .base_agent import BaseAgent
from src.ingestion.vector_store import FlightVectorStore, VectorStore

class VectorAgent(BaseAgent):
    """
    Autonomous document retrieval agent.
    Decides dynamically if vector search is needed and retrieves docs.
    """
    def __init__(self, vector_store: FlightVectorStore = None):
        self.vector_store = vector_store or FlightVectorStore()

    def can_handle(self, payload: Dict[str, Any]) -> bool:
        # Run if vector search has not been performed yet and intent is not 'flight_id_lookup'
        intent_type = payload.get("intent", {}).get("type")
        return "retrieved_docs" not in payload and intent_type != "flight_id_lookup"

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        query = payload.get("query", "")

        results = self.vector_store.query_with_scores(query)
        if not results:
            # No results found
            payload["retrieved_docs"] = []
            payload["distances"] = []
            payload["metadatas"] = []
            return payload

        docs, distances, metadatas = zip(*results)
        payload["retrieved_docs"] = list(docs)
        payload["distances"] = list(distances)
        payload["metadatas"] = list(metadatas)
        return payload