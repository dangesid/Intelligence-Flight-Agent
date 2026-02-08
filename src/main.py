from src.orchestrator.orchestrator import Orchestrator

# Agents
from src.agents.query_intent_agent import QueryIntentAgent
from src.agents.vector_agent import VectorAgent
from src.agents.knowledge_gap_agent import KnowledgeGapAgent
from src.agents.clara_agent import ClaraAgent

from src.llm_engine.ollama_client import OllamaWrapper

def main():
    query = input("Ask a flight question: ").strip()

    #Initialise LLM Client 
    llm_client = OllamaWrapper()

    # 🧠 Initial shared memory
    payload = {
        "query": query
    }

    # 🤖 Register agents (order does NOT matter)
    agents = [
        QueryIntentAgent(llm_client),
        VectorAgent(),
        KnowledgeGapAgent(),
        ClaraAgent()
    ]

    orchestrator = Orchestrator(agents)
    result = orchestrator.run(payload)

    print("\n🧠 Answer:")
    answer = result.get("answer", {})
    print(answer.get("text", "No answer produced."))

    print(f"\n🔐 Confidence: {answer.get('confidence', 0.0)}")

    print("\n🧬 Orchestration Trace:")
    print(result.get("orchestration"))


if __name__ == "__main__":
    main()
