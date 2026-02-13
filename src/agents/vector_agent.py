# src/agents/vector_agent.py
from typing import Dict, Any, List
from src.agents.base_agent import BaseAgent
from src.vector_store import FlightVectorStore

class VectorAgent(BaseAgent):
    """
    Autonomous Vector Retrieval Agent
    """

    def __init__(
        self,
        vector_store: FlightVectorStore | None = None,
        threshold: float = 0.6,  # slightly lower
        top_n_fallback: int = 5,
    ):
        self.vector_store = vector_store or FlightVectorStore()
        self.threshold = threshold
        self.top_n_fallback = top_n_fallback

    def can_handle(self, payload: Dict[str, Any]) -> bool:
        return "query" in payload and "retrieved_docs" not in payload

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

        # Filter by threshold
        filtered_results = [(doc, dist, meta) for doc, dist, meta in results if dist <= self.threshold]

        # Fallback to top N
        if not filtered_results:
            filtered_results = results[:self.top_n_fallback]

        docs, distances, metadatas = zip(*filtered_results)

        payload["retrieved_docs"] = list(docs)
        payload["distances"] = list(distances)
        payload["metadatas"] = list(metadatas)

        print(f"\n🔍 [VectorAgent] Retrieved Documents (threshold={self.threshold}):")
        for i, (doc, dist) in enumerate(zip(docs, distances), start=1):
            print(f"[Doc {i}] Distance={dist:.3f} | {doc}")

        print(f"Distances of all retrieved docs: {tuple(distances)}")
        return payload
