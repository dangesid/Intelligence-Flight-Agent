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
from src.llm_clients.azure_openai_client import AzureOpenAIWrapper

# 📦 Persistent system memory
STATE_FILE = Path("system_state.json")


def load_state() -> list:
    if not STATE_FILE.exists():
        return []

    try:
        content = STATE_FILE.read_text().strip()
        if not content:
            return []
        data = json.loads(content)
        # Convert old dict format to list
        if isinstance(data, dict):
            return data.get("knowledge_gaps", [])
        elif isinstance(data, list):
            return data
        else:
            return []
    except Exception:
        return []


def save_state(state: list) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


def main():
    query = input("Ask a question: ").strip()

    llm_client = AzureOpenAIWrapper()
    print(f"✅ Using LLM: {llm_client.__class__.__name__}")

    # ✅ Load persistent memory
    persistent_state = load_state()

    # ✅ Fresh execution payload
    payload = {
        "query": query,
        "knowledge_gaps": []
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

    # Safely extract gap info
    gap_signal = result.get("gap_signal") or {}
    missing_knowledge = result.get("missing_knowledge")
    llm_analysis = result.get("llm_analysis") or {}

    # ✅ Build structured entry for system_state.json
    entry = {
        "question": query,
        "system_answer": result.get("answer", {}).get("text", "No answer produced."),
        "gap_reason": gap_signal.get("reason") if gap_signal else None,
        "severity": gap_signal.get("severity") if gap_signal else None,
        "missing_knowledge": missing_knowledge,
        "llm_analysis": llm_analysis
    }

    # Append to persistent state
    persistent_state.append(entry)
    save_state(persistent_state)

    # 🧠 Answer
    print("\n🧠 Answer:")
    print(entry["system_answer"])
    print(f"\n🔐 Confidence: {result.get('answer', {}).get('confidence', 0.0)}")

    # 🧩 Knowledge gap signal (only if real gap exists)
    if gap_signal and gap_signal.get("severity") != "low":
        print("\n🧩 Knowledge Gap Detected:")
        print(f"Reason       : {gap_signal.get('reason')}")
        print(f"Severity     : {gap_signal.get('severity')}")
        if missing_knowledge:
            print(f"Missing Info : {missing_knowledge}")

    # 🧬 Orchestration trace (optional)
    if result.get("orchestration"):
        print("\n🧬 Orchestration Trace (debug only):")
        print(result["orchestration"])


if __name__ == "__main__":
    main()
