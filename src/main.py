from src.ingestion.vector_store import FlightVectorStore
from src.llm_engine.factory import get_llm
from src.evaluation.gap_analyzer import KnowledgeGapAnalyzer
from src.agents.query_intent_agent import QueryIntentAgents

RELEVANCE_THRESHOLD = 0.60

# def is_relevant(distances):
#     assert all(isinstance(d, float) for d in distances), "Distances must be floats"
#     return any(d <= RELEVANCE_THRESHOLD for d in distances)

def relevance_bucket(best_distance: float):
    if best_distance <= 0.75:
        return "HIGH"
    elif best_distance <= 0.85:
        return "MEDIUM"
    else:
        return "LOW"


if __name__ == "__main__":
    vector_store = FlightVectorStore()
    llm = get_llm()
    gap_analyzer = KnowledgeGapAnalyzer()

    query = input("Ask a flight question: ")

    results = vector_store.query_with_scores(query)
    intent_agent = QueryIntentAgents()
    intent = intent_agent.classify(query)

    docs, distances, metadatas = zip(*results)
    domain_summary = vector_store.inspect_domain()

    print("\n🔎 Retrieval diagnostics:")
    for d, doc in zip(distances, docs):
        print(f"Distance: {d:.3f} | {doc[:80]}...")

    best_distance = min(distances)
    bucket = relevance_bucket(best_distance)

    print(f"\n🧪 Relevance bucket: {bucket}")
    print(f"📉 Best distance: {best_distance:.3f}")

    if bucket == "LOW":
        print("\n❌ No confident answer possible.")

        explanation = gap_analyzer.analyze(
            query=query,
            intent=intent,
            distance=list(distances),
            documents=list(docs),
            metadatas=list(metadatas),
            domain_summary=domain_summary
        )

        print("\n🧠 Why this failed:")
        print(explanation)
        print("\n🧠 Answer:")
        print("I don’t have enough reliable information to answer this question.")
   
    elif bucket == "MEDIUM":
        print("\n⚠️ Partial match found. Answering cautiously.")

        answer = llm.generate_with_context(
            query,
            list(docs),
            instruction="Answer carefully using only the given data. "
                        "If information is missing, say so explicitly."
        )

        print("\n🧠 Answer:\n", answer)
    else:  # HIGH
        answer = llm.generate_with_context(query, list(docs))
        print("\n🧠 Answer:\n", answer)