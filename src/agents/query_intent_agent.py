# src/agents/query_intent_agent.py
from typing import Dict, Any, Optional
from langchain_core.messages import HumanMessage
from src.llm_clients.azure_openai_client import AzureOpenAIWrapper
from src.agents.base_agent import BaseAgent
import json


class QueryIntentAgent(BaseAgent):
    """
    Agentic Query Intent Extraction
    - Autonomous evaluation of whether it should act
    - Can propose actions (currently none, but structure exists for tools/next agents)
    - Execute method replaces legacy run()
    """

    def __init__(self):
        super().__init__(name="QueryIntentAgent")
        self.llm = AzureOpenAIWrapper()

    # -------------------
    # LEGACY METHODS
    # -------------------
    def can_handle(self, payload: Dict[str, Any]) -> bool:
        has_query = bool(payload.get("query"))
        no_intent = not payload.get("intent")
        return has_query and no_intent

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.execute(payload)

    # -------------------
    # AGENTIC METHODS
    # -------------------
    def evaluate(self, payload: Dict[str, Any]) -> bool:
        """
        Decide if this agent should act.
        Uses same logic as legacy can_handle, can extend with more intelligence later.
        """
        return self.can_handle(payload)

    def propose_action(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Propose next steps or tools.
        Currently no extra tools, but structure is here for agentic orchestration.
        """
        return {
            "next_agents": ["VectorAgent"],  # Suggest VectorAgent after intent extraction
            "tools_to_call": [],             # Could add LLM tools if needed
            "modify_payload": {}             # No modifications yet
        }

    def execute(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        query = payload.get("query", "")
        
        if not query:
            return payload  # Nothing to do

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