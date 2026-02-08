from typing import Dict, Any
from src.agents.base_agent import BaseAgent
from src.ingestion.vector_store import FlightVectorStore


class RetrievalAgent(BaseAgent):
    """
    Performs semantic retrieval when needed.
    """

    def __init__(self):
        self.vector_store = FlightVectorStore()

    def can_handle(self, payload: Dict[str, Any]) -> bool:
        return (
            payload.get("intent", {}).get("type") == "semantic_search"
            and "retrieved_docs" not in payload
        )

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        query = payload["query"]

        results = self.vector_store.query_with_scores(query)

        if not results:
            payload["retrieved_docs"] = []
            payload["distances"] = []
            payload["metadatas"] = []
            return payload

        docs, distances, metadatas = zip(*results)

        payload["retrieved_docs"] = list(docs)
        payload["distances"] = list(distances)
        payload["metadatas"] = list(metadatas)

        return payload
