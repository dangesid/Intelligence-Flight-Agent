# src/main.py
import json
import os
from pathlib import Path
from typing import Dict, Any

# Agents
from src.agents.query_intent_agent import QueryIntentAgent
from src.agents.vector_agent import VectorAgent
from src.agents.context_builder_agent import ContextBuilderAgent
from src.agents.knowledge_gap_agent import KnowledgeGapAgent
from src.agents.coverage_evaluator_agent import CoverageEvaluatorAgent
from src.agents.feedback_memory_agents import FeedbackMemoryAgent
from src.agents.ingestion_planner_agent import IngestionPlannerAgent
from src.agents.clara_agent import ClaraAgent
from src.agents.final_answer_agent import FinalizerAgent

# Orchestrator
from src.orchestrator.orchestrator import Orchestrator

# Vector Store
from src.vector_store import FlightVectorStore
from src.config import settings

# ==================================================
# 🔍 LLM Configuration Logger
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
# 📦 Persistent System State (Memory + Active DB)
# ==================================================

STATE_FILE = Path("system_state.json")


def load_state() -> Dict[str, Any]:
    """Load persistent system state safely and handle old formats."""
    if not STATE_FILE.exists():
        return {
            "knowledge_gaps": [],
            "active_vector_source": "COSMOS"
        }

    try:
        content = STATE_FILE.read_text().strip()
        if not content:
            return {
                "knowledge_gaps": [],
                "active_vector_source": "COSMOS"
            }

        state = json.loads(content)

        # Convert old list format to dict
        if isinstance(state, list):
            state = {"knowledge_gaps": state}

        if "knowledge_gaps" not in state:
            state["knowledge_gaps"] = []

        if "active_vector_source" not in state:
            state["active_vector_source"] = "COSMOS"

        return state

    except Exception:
        return {
            "knowledge_gaps": [],
            "active_vector_source": "COSMOS"
        }


def save_state(state: Dict[str, Any]):
    """Save persistent state safely."""
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ==================================================
# 🧠 Determine Active Vector DB
# ==================================================

def get_active_vector_path(state: Dict[str, Any]) -> str:
    source = state.get("active_vector_source", "COSMOS")

    if source == "COSMOS":
        print("\n🌌 SYSTEM ACTIVE DB: COSMOS (Primary)\n")
        return settings.VECTOR_DB_PATH_COSMOS

    elif source == "CSV":
        print("\n📄 SYSTEM ACTIVE DB: CSV (Fallback)\n")
        return settings.VECTOR_DB_PATH_CSV

    print("\n⚠️ Unknown DB source. Defaulting to COSMOS.\n")
    return settings.VECTOR_DB_PATH_COSMOS


# ==================================================
# 🚀 MAIN
# ==================================================

def main():
    log_llm_configuration()

    query = input("Ask a question: ").strip()

    # Load persistent state
    persistent_state = load_state()

    # Determine correct vector DB path
    vector_path = get_active_vector_path(persistent_state)

    # Shared vector store (DO NOT reset in runtime)
    vector_store = FlightVectorStore(
        vector_db_path=vector_path,
        reset_collection=False
    )

    # Payload
    payload = {
        "query": query,
        "knowledge_gaps": persistent_state.get("knowledge_gaps", [])
    }

    # Initialize agents
    agents = [
        QueryIntentAgent(),
        VectorAgent(vector_store=vector_store),
        ContextBuilderAgent(),
        KnowledgeGapAgent(),
        CoverageEvaluatorAgent(),
        FeedbackMemoryAgent(state_path=str(STATE_FILE)),
        IngestionPlannerAgent(),
        ClaraAgent(),
        FinalizerAgent()
    ]

    # Run orchestrator
    orchestrator = Orchestrator(agents)
    result = orchestrator.run(payload)

    # Persist long-term memory + active DB source
    updated_state = {
        "knowledge_gaps": result.get("knowledge_gaps", []),
        "active_vector_source": persistent_state.get(
            "active_vector_source",
            "COSMOS"
        )
    }

    save_state(updated_state)

    # 🧠 Final Answer
    answer = result.get("answer", {})
    print("\n🧠 Answer:")
    print(answer.get("text", "No answer produced."))
    print(f"\n🔐 Confidence: {answer.get('confidence', 0.0)}")

    # 🧩 Knowledge gap signal
    if result.get("gap_signal"):
        print("\n🧩 Knowledge Gap Signal:")
        print(result["gap_signal"])

    # 🧬 Orchestration trace
    print("\n🧬 Orchestration Trace:")
    print(result.get("orchestration"))


if __name__ == "__main__":
    main()