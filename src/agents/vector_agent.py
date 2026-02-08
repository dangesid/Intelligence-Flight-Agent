from typing import Dict, Any, List
from src.agents.base_agent import BaseAgent
from src.vector_store import FlightVectorStore


class VectorAgent(BaseAgent):
    """
    Autonomous Vector Retrieval Agent

    Responsibilities:
    - Decide if vector retrieval is needed
    - Perform semantic search
    - Populate payload with retrieved docs, distances, metadata
    - Make NO decisions about relevance or answers
    """

    def __init__(self, vector_store: FlightVectorStore | None = None):
        self.vector_store = vector_store or FlightVectorStore()

    def can_handle(self, payload: Dict[str, Any]) -> bool:
        """
        Run if:
        - A query exists
        - Retrieval has not already been performed
        """
        return (
            "query" in payload
            and "retrieved_docs" not in payload
        )

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        query: str = payload.get("query", "").strip()

        if not query:
            payload["retrieved_docs"] = []
            payload["distances"] = []
            payload["metadatas"] = []
            return payload

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

        print("\n🔍 [VectorAgent] Retrieved Documents:")
        for i, (doc, dist) in enumerate(zip(docs, distances), start=1):
            print(f"[Doc {i}] Distance={dist:.3f} | {doc}")

        return payload
