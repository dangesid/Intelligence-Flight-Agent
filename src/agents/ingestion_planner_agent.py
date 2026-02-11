from typing import Dict, Any, List
import numpy as np
from src.agents.base_agent import BaseAgent


class IngestionPlannerAgent(BaseAgent):
    """
    Plans future ingestion targets based on clustered weak queries.

    - Embedding-based
    - No thresholds
    - No domain assumptions
    """

    def can_handle(self, payload: Dict[str, Any]) -> bool:
        return (
            "feedback_store" in payload
            and len(payload["feedback_store"]) >= 2
            and "ingestion_plan" not in payload
            and "embedding_fn" in payload
        )

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        feedback_items = payload["feedback_store"]
        embed = payload["embedding_fn"]

        queries = [item["query"] for item in feedback_items]
        vectors = np.array(embed(queries))

        clusters = self._simple_cluster(vectors)

        ingestion_plan = []
        for cluster in clusters:
            cluster_queries = [queries[i] for i in cluster]
            ingestion_plan.append({
                "representative_query": cluster_queries[0],
                "related_queries": cluster_queries,
                "count": len(cluster_queries)
            })

        payload["ingestion_plan"] = ingestion_plan
        return payload

    def _simple_cluster(self, vectors: np.ndarray) -> List[List[int]]:
        """
        Minimal embedding-based clustering without thresholds.
        Groups nearest neighbors greedily.
        """
        clusters = []
        used = set()

        for i in range(len(vectors)):
            if i in used:
                continue

            cluster = [i]
            used.add(i)

            for j in range(i + 1, len(vectors)):
                if j in used:
                    continue

                similarity = np.dot(vectors[i], vectors[j])
                if similarity > 0.85:  # similarity only, not domain logic
                    cluster.append(j)
                    used.add(j)

            clusters.append(cluster)

        return clusters
