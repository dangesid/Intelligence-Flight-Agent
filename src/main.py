import json
from pathlib import Path

from src.agents.context_builder_agent import ContextBuilderAgent
from src.orchestrator.orchestrator import Orchestrator

# Agents
from src.agents.query_intent_agent import QueryIntentAgent
from src.agents.vector_agent import VectorAgent
from src.agents.knowledge_gap_agent import KnowledgeGapAgent
from src.agents.clara_agent import ClaraAgent
from src.agents.feedback_memory_agents import FeedbackMemoryAgent
from src.agents.ingestion_planner_agent import IngestionPlannerAgent
from src.agents.coverage_evaluator_agent import CoverageEvaluatorAgent
from src.agents.final_answer_agent import FinalizerAgent

from src.llm_engine.ollama_client import OllamaWrapper

# 📦 Persistent system memory
STATE_FILE = Path("system_state.json")


def load_state() -> dict:
    if not STATE_FILE.exists():
        return {}

    try:
        content = STATE_FILE.read_text().strip()
        if not content:
            return {}
        return json.loads(content)
    except Exception:
        return {}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


def main():
    query = input("Ask a question: ").strip()

    llm_client = OllamaWrapper()

    # ✅ Load only long-term memory
    persistent_state = load_state()

    # ✅ Fresh payload per execution
    payload = {
        "query": query,
        "knowledge_gaps": persistent_state.get("knowledge_gaps", [])
    }

    agents = [
        QueryIntentAgent(llm_client),
        VectorAgent(),
        ContextBuilderAgent(),
        KnowledgeGapAgent(llm_client),
        CoverageEvaluatorAgent(),
        IngestionPlannerAgent(),
        ClaraAgent(),
        FinalizerAgent(),
        FeedbackMemoryAgent(
            llm_client=llm_client,
            state_path=str(STATE_FILE)
        )
    ]

    orchestrator = Orchestrator(agents)
    result = orchestrator.run(payload)

    # ✅ Persist ONLY long-term memory
    memory_to_save = {
        "knowledge_gaps": result.get("knowledge_gaps", [])
    }

    save_state(memory_to_save)

    answer = result.get("answer", {})


    # 🧠 Answer
    print("\n🧠 Answer:")
    answer = result.get("answer", {})
    print(answer.get("text", "No answer produced."))
    print(f"\n🔐 Confidence: {answer.get('confidence', 0.0)}")

    # 🧩 Knowledge gap signal
    if "gap_signal" in result:
        gap = result["gap_signal"]
        print("\n🧩 Knowledge Gap Detected:")
        print(f"Reason       : {gap.get('reason')}")
        print(f"Severity     : {gap.get('severity')}")
        if result.get("missing_knowledge"):
            print(f"Missing Info : {result.get('missing_knowledge')}")

    # Optional: Orchestration trace for debugging
    if result.get("orchestration"):
        print("\n🧬 Orchestration Trace (debug only):")
        print(result["orchestration"])


if __name__ == "__main__":
    main()
