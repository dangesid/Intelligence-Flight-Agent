# src/agents/query_intent_agent.py
from typing import Dict, Any
from langchain_core.messages import HumanMessage
from src.llm_clients.azure_openai_client import AzureOpenAIWrapper
import json


class QueryIntentAgent:
    def __init__(self):  # No parameters - creates its own LLM
        self.llm = AzureOpenAIWrapper()

    def can_handle(self, payload: Dict[str, Any]) -> bool:
        has_query = bool(payload.get("query"))
        no_intent = not payload.get("intent")
        return has_query and no_intent

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        query = payload["query"]
        
        try:
            prompt = (
                "Extract the intent from this flight query. "
                "Return ONLY a JSON object with keys: source, destination, date, preferences.\n"
                f"Query: {query}"
            )
            
            response = self.llm.generate(messages=[HumanMessage(content=prompt)])
            intent_text = response["choices"][0]["message"]["content"]
            intent = json.loads(intent_text.strip())
            
            payload["intent"] = intent
            print(f"✅ Extracted intent: {intent}")
            
        except Exception as e:
            print(f"⚠️ QueryIntentAgent Azure GPT call failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback: basic extraction
            payload["intent"] = {
                "source": "PNQ" if "PUNE" in query.upper() else None,
                "destination": None,
                "date": None,
                "preferences": {}
            }
        
        return payload