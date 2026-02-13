# src/main.py
import json
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

# 📦 Persistent system memory
STATE_FILE = Path("system_state.json")


def load_state() -> Dict[str, Any]:
    """Load persistent system state safely and handle old formats."""
    if not STATE_FILE.exists():
        return {"knowledge_gaps": []}

    try:
        content = STATE_FILE.read_text().strip()
        if not content:
            return {"knowledge_gaps": []}

        state = json.loads(content)

        # Convert old list format to dict
        if isinstance(state, list):
            state = {"knowledge_gaps": state}

        # Ensure key exists
        if "knowledge_gaps" not in state or not isinstance(state["knowledge_gaps"], list):
            state["knowledge_gaps"] = []

        return state
    except Exception:
        return {"knowledge_gaps": []}


def save_state(state: Dict[str, Any]):
    """Save persistent state to disk safely."""
    STATE_FILE.write_text(json.dumps(state, indent=2))


def main():
    query = input("Ask a question: ").strip()

    # ✅ Load persistent state safely
    persistent_state = load_state()

    # ✅ Fresh payload per execution
    payload = {
        "query": query,
        "knowledge_gaps": persistent_state.get("knowledge_gaps", [])
    }

    # ✅ Initialize shared vector store once (efficient)
    vector_store = FlightVectorStore()

    # Initialize agents - each agent creates its own dependencies internally
    agents = [
        QueryIntentAgent(),  # Creates its own AzureOpenAIWrapper internally
        VectorAgent(vector_store=vector_store),  # Pass shared vector store
        ContextBuilderAgent(),
        KnowledgeGapAgent(),  # Creates its own AzureOpenAIWrapper internally
        CoverageEvaluatorAgent(),
        FeedbackMemoryAgent(state_path=str(STATE_FILE)),  # Only takes state_path
        IngestionPlannerAgent(),
        ClaraAgent(),  # Creates its own AzureOpenAIWrapper internally
        FinalizerAgent()
    ]

    # Run orchestrator
    orchestrator = Orchestrator(agents)
    result = orchestrator.run(payload)

    # ✅ Persist ONLY long-term memory safely
    memory_to_save = {
        "knowledge_gaps": result.get("knowledge_gaps", [])
    }
    save_state(memory_to_save)

    # 🧠 Answer
    answer = result.get("answer", {})
    print("\n🧠 Answer:")
    print(answer.get("text", "No answer produced."))
    print(f"\n🔐 Confidence: {answer.get('confidence', 0.0)}")

    # 🧩 Knowledge gap signal
    if "gap_signal" in result and result["gap_signal"]:
        print("\n🧩 Knowledge Gap Signal:")
        print(result["gap_signal"])

    # 🧬 Orchestration trace
    print("\n🧬 Orchestration Trace:")
    print(result.get("orchestration"))


if __name__ == "__main__":
    main()