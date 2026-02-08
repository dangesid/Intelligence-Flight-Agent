from typing import Dict, Any

from src.ingestion.vector_store import FlightVectorStore
from src.agents.query_intent_agent import QueryIntentAgent
from src.agents.knowledge_gap_agent import KnowledgeGapAgent
from src.reasoning.clara_agent import ClaraAgent


# --------------------------------------------------
# Relevance bucketing
# --------------------------------------------------
def relevance_bucket(best_distance: float) -> str:
    if best_distance <= 0.75:
        return "HIGH"
    elif best_distance <= 0.85:
        return "MEDIUM"
    else:
        return "LOW"


# --------------------------------------------------
# Broad vs specific question detector
# --------------------------------------------------
def is_broad_flight_query(query: str) -> bool:
    q = query.lower().strip()
    return (
        "flight" in q
        and not any(char.isdigit() for char in q)
        and "from" not in q
        and "to" not in q
        and len(q.split()) <= 7
    )


# --------------------------------------------------
# Confidence calibration
# --------------------------------------------------
def confidence_from_distance(distance: float) -> float:
    if distance >= 1.05:
        return 0.0
    return round(max(0.0, 1.0 - (distance - 0.6)), 2)


# ==================================================
# MAIN
# ==================================================
if __name__ == "__main__":

    vector_store = FlightVectorStore()

    intent_agent = QueryIntentAgent()
    gap_agent = KnowledgeGapAgent()
    clara_agent = ClaraAgent()

    query = input("Ask a flight question: ").strip()

    # ==================================================
    # SHARED PAYLOAD (AGENT MEMORY)
    # ==================================================
    payload: Dict[str, Any] = {
        "query": query
    }

    # -------------------------
    # INTENT CLASSIFICATION
    # -------------------------
    intent_payload = intent_agent.classify(query)
    payload["intent"] = intent_payload

    intent = intent_payload.get("intent")

    # =====================================================
    # FLIGHT ID LOOKUP (DIRECT FACT PATH)
    # =====================================================
    if intent == "flight_id_lookup":
        flight_number = intent_payload.get("flight_number")
        result = vector_store.lookup_by_flight_id(flight_number)

        if not result:
            payload["gap_report"] = {
                "reason": "Flight ID not found",
                "flight_number": flight_number
            }
        else:
            payload["retrieved_docs"] = [
                (
                    f"Flight {result['flight']} operates from {result['origin']} "
                    f"to {result['dest']}. Departs at {result['sched_dep_time']} "
                    f"and arrives at {result['sched_arr_time']}."
                )
            ]

        payload = clara_agent.run(payload)

        print("\n🧠 Answer:")
        print(payload["answer"]["text"])
        print(f"\n🔐 Confidence: {payload['answer']['confidence']:.2f}")
        exit()

    # =====================================================
    # VECTOR SEARCH (PRIMARY PATH)
    # =====================================================
    results = vector_store.query_with_scores(query)

    if not results:
        payload["gap_report"] = gap_agent.analyze(
            query=query,
            intent=intent_payload,
            distances=[],
            metadatas=[],
            domain_summary=vector_store.inspect_domain()
        )

        payload = clara_agent.run(payload)

        print("\n🧠 Answer:")
        print(payload["answer"]["text"])
        print(f"\n🔐 Confidence: {payload['answer']['confidence']:.2f}")
        exit()

    docs, distances, metadatas = zip(*results)

    payload["retrieved_docs"] = list(docs)
    payload["distances"] = list(distances)
    payload["metadatas"] = list(metadatas)

    # -------------------------
    # DEBUG / TRACE
    # -------------------------
    print("\n🔎 Retrieval diagnostics:")
    for d, doc in zip(distances, docs):
        print(f"Distance: {d:.3f} | {doc[:90]}...")

    best_distance = min(distances)
    bucket = relevance_bucket(best_distance)
    confidence = confidence_from_distance(best_distance)

    payload["confidence_hint"] = confidence
    payload["relevance_bucket"] = bucket
    payload["broad_query"] = is_broad_flight_query(query)

    print(f"\n🧪 Relevance bucket: {bucket}")
    print(f"📉 Best distance: {best_distance:.3f}")

    # =====================================================
    # GAP ANALYSIS (LOW RELEVANCE)
    # =====================================================
    if bucket == "LOW":
        payload["gap_report"] = gap_agent.analyze(
            query=query,
            intent=intent_payload,
            distances=list(distances),
            metadatas=list(metadatas),
            domain_summary=vector_store.inspect_domain()
        )

    # =====================================================
    # FINAL ANSWER → CLARA AGENT
    # =====================================================
    payload = clara_agent.run(payload)

    print("\n🧠 Answer:")
    print(payload["answer"]["text"])
    print(f"\n🔐 Confidence: {payload['answer']['confidence']:.2f}")
