from src.agents.base_agent import BaseAgent
from src.llm_engine.ollama_client import OllamaWrapper

class QueryIntentAgent(BaseAgent):
    """
    Uses LLM to classify intent dynamically. Fully agentic.
    """
    def __init__(self, llm_client: OllamaWrapper):
        self.llm = llm_client

    def can_handle(self, payload):
        return "query" in payload and "intent" not in payload

    def run(self, payload):
        query = payload["query"]
        prompt = f"""
        You are an Intent Classifier Agent. 
        Classify this user query into one of the following intents:
        simple_search, gap_analysis, complex_reasoning, other
        Respond with only the label.
        Query: "{query}"
        """
        intent = self.llm.generate(prompt)
        payload["intent"] = {"type": intent.strip()}
        return payload