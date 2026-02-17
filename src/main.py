# src/main.py

import json
from pathlib import Path
from typing import Dict, Any

# Agents
from src.agents.query_intent_agent import QueryIntentAgent
from src.agents.vector_agent import VectorAgent
from src.agents.context_builder_agent import ContextBuilderAgent
from src.agents.knowledge_gap_agent import KnowledgeGapAgent
from src.agents.feedback_memory_agents import FeedbackMemoryAgent
from src.agents.clara_agent import ClaraAgent
from src.agents.final_answer_agent import FinalizerAgent

from src.orchestrator.orchestrator import build_graph
from src.vector_store import FlightVectorStore
from src.config import settings


# ==================================================
# LLM CONFIG LOGGER
# ==================================================
def log_llm_configuration():
    provider = settings.LLM_PROVIDER.lower()
    print("\n🤖 LLM CONFIGURATION")

    if provider == "azure":
        print("Provider: AZURE OPENAI")
        print(f"Endpoint: {settings.AZURE_OPENAI_ENDPOINT}")
        print(f"Deployment: {settings.AZURE_OPENAI_DEPLOYMENT}")
        print(f"API Version: {settings.AZURE_OPENAI_API_VERSION}")
    else:
        print(f"Provider: {provider.upper()}")
    print("")


# ==================================================
# STATE FILE
# ==================================================
STATE_FILE = Path("system_state.json")


def load_state() -> Dict[str, Any]:
    if not STATE_FILE.exists():
        return {"knowledge_gaps": [], "active_vector_source": "COSMOS"}
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {"knowledge_gaps": [], "active_vector_source": "COSMOS"}


def save_state(state: Dict[str, Any]):
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ==================================================
# ACTIVE DB
# ==================================================
def get_active_vector_path(state: Dict[str, Any]) -> str:
    source = state.get("active_vector_source", "COSMOS")
    if source == "COSMOS":
        print("\n🌌 SYSTEM ACTIVE DB: COSMOS (Primary)\n")
        return settings.VECTOR_DB_PATH_COSMOS
    elif source == "CSV":
        print("\n📄 SYSTEM ACTIVE DB: CSV (Fallback)\n")
        return settings.VECTOR_DB_PATH_CSV
    return settings.VECTOR_DB_PATH_COSMOS


# ==================================================
# MAIN
# ==================================================
def main():

    log_llm_configuration()

    query = input("Ask a question: ").strip()

    persistent_state = load_state()
    vector_path = get_active_vector_path(persistent_state)

    vector_store = FlightVectorStore(
        vector_db_path=vector_path,
        reset_collection=False
    )

    payload = {
        "query": query,
        "knowledge_gaps": persistent_state.get("knowledge_gaps", [])
    }

    # -------------------------
    # AGENTIC AGENTS
    # -------------------------
    agents = [
        QueryIntentAgent(),
        VectorAgent(vector_store=vector_store),
        ContextBuilderAgent(),
        KnowledgeGapAgent(),
        FeedbackMemoryAgent(state_path=str(STATE_FILE)),
        ClaraAgent(),
        FinalizerAgent()
    ]

    # Build the agentic LangGraph pipeline
    app = build_graph(agents)

    # Invoke the graph (dynamically evaluates agents)
    result = app.invoke(payload)

    # -------------------------
    # SAVE UPDATED STATE
    # -------------------------
    updated_state = {
        "knowledge_gaps": result.get("knowledge_gaps", []),
        "active_vector_source": persistent_state.get(
            "active_vector_source", "COSMOS"
        )
    }
    save_state(updated_state)

    # -------------------------
    # PRINT FINAL OUTPUT
    # -------------------------
    answer = result.get("answer", {})
    answer_text = answer.get("text", "No answer produced.")
    confidence = answer.get("confidence", 0.0)

    print(f"\n🧠 Answer:\n{answer_text}")
    print(f"\n🔐 Confidence: {confidence}")

    gap_signal = result.get("gap_signal")
    if gap_signal:
        print("\n🧩 Knowledge Gap Signal (LLM Explanation):")
        print(gap_signal.get("llm_analysis", {}).get("knowledge_gap"))
        print("Severity:", gap_signal.get("severity"))


if __name__ == "__main__":
    main()
