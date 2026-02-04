from src.ingestion.vector_store import FlightVectorStore
from src.agents.query_intent_agent import QueryIntentAgents
from src.agents.knowledge_gap_agent import KnowledgeGapAgent
from src.reasoning.clara_system import ClaraSystem


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
# Broad vs specific question detector (semantic-lite)
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
    intent_agent = QueryIntentAgents()
    gap_agent = KnowledgeGapAgent()
    clara = ClaraSystem()

    query = input("Ask a flight question: ").strip()

    intent_payload = intent_agent.classify(query)
    intent = intent_payload.get("intent")

    # =====================================================
    # FLIGHT ID LOOKUP (STRICT, NO VECTOR)
    # =====================================================
    if intent == "flight_id_lookup":
        flight_number = intent_payload.get("flight_number")
        result = vector_store.lookup_by_flight_id(flight_number)

        print("\n🧠 Answer:")
        if not result:
            print(f"No flight with ID {flight_number} exists.")
            print("\n🔐 Confidence: 0.90")
            exit()

        print(
            f"Flight {result['flight']} operates from {result['origin']} "
            f"to {result['dest']}. Departs at {result['sched_dep_time']} "
            f"and arrives at {result['sched_arr_time']}."
        )
        print("\n🔐 Confidence: 0.95")
        exit()

    # =====================================================
    # VECTOR SEARCH (PRIMARY PATH)
    # =====================================================
    results = vector_store.query_with_scores(query)

    if not results:
        print("\n🧠 Answer:")
        print("I don’t have relevant flight data for this query.")
        print("\n🔐 Confidence: 0.30")
        exit()

    docs, distances, metadatas = zip(*results)

    print("\n🔎 Retrieval diagnostics:")
    for d, doc in zip(distances, docs):
        print(f"Distance: {d:.3f} | {doc[:80]}...")

    best_distance = min(distances)
    bucket = relevance_bucket(best_distance)
    confidence = confidence_from_distance(best_distance)

    print(f"\n🧪 Relevance bucket: {bucket}")
    print(f"📉 Best distance: {best_distance:.3f}")

    # =====================================================
    # 🟡 BROAD FLIGHT QUESTIONS → SUMMARIZE RESULTS
    # =====================================================
    if is_broad_flight_query(query):
        response = clara.answer(
            query=query,
            docs=list(docs),
            confidence=confidence,
            cautious=False
        )

        print("\n🧠 Answer:")
        print(response["answer"])
        print(f"\n🔐 Confidence: {response['confidence']:.2f}")
        exit()

    # =====================================================
    # 🟢 SPECIFIC QUESTIONS (STRICT GATING)
    # =====================================================
    if bucket in ("HIGH", "MEDIUM"):
        response = clara.answer(
            query=query,
            docs=list(docs),
            confidence=confidence,
            cautious=(bucket == "MEDIUM")
        )
    else:
        gap_report = gap_agent.analyze(
            query=query,
            intent=intent_payload,
            distances=list(distances),
            metadatas=list(metadatas),
            domain_summary=vector_store.inspect_domain()
        )

        response = clara.no_answer(
            gap_report=gap_report,
            confidence=0.20
        )

    print("\n🧠 Answer:")
    print(response["answer"])
    print(f"\n🔐 Confidence: {response['confidence']:.2f}")
