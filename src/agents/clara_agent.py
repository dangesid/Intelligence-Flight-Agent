# src/agents/clara_agent.py
from typing import Dict, Any, Optional
from langchain_core.messages import HumanMessage
from src.llm_clients.azure_openai_client import AzureOpenAIWrapper
from src.agents.base_agent import BaseAgent


class ClaraAgent(BaseAgent):
    """
    Agentic CLARA Reasoning Agent
    - Evaluates autonomously whether it should act
    - Can propose next steps/tools (currently none, structure ready for expansion)
    - Executes reasoning over context to generate final answer
    """

    def __init__(self):
        super().__init__(name="ClaraAgent")
        self.llm = AzureOpenAIWrapper()

    # -------------------
    # LEGACY METHODS
    # -------------------
    def can_handle(self, payload: Dict[str, Any]) -> bool:
        """
        Handles the query when:
        - Context has been retrieved
        - Query hasn't been finalized
        """
        has_context = bool(payload.get("context"))
        not_finalized = not payload.get("finalized", False)
        return has_context and not_finalized

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.execute(payload)

    # -------------------
    # AGENTIC METHODS
    # -------------------
    def evaluate(self, payload: Dict[str, Any]) -> bool:
        """
        Decide if ClaraAgent should act
        """
        return self.can_handle(payload)

    def propose_action(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        ClaraAgent doesn't need to call other agents/tools for now
        but structure is ready for agentic orchestration
        """
        return {
            "next_agents": ["FinalizerAgent"],  # Suggest FinalizerAgent after reasoning
            "tools_to_call": [],                # Could add tools in the future
            "modify_payload": {}                # No modifications for now
        }

    def execute(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        query = payload.get("query", "")
        context = payload.get("context", "")

        if not context:
            reason = (
                "The system does not have any documents matching your query. "
                "No relevant flight information is available in the vector store."
            )
            payload["answer"] = {"text": reason, "confidence": 0.7}
            payload["finalized"] = True
            return payload

        try:
            prompt = (
                "You are a helpful flight assistant. "
                "Answer strictly using the provided context. "
                "If the information is not present, say 'Data not available'. "
                "Do NOT use any external knowledge or make assumptions.\n\n"
                f"Context:\n{context}\n\nQuestion:\n{query}"
            )

            response = self.llm.generate(messages=[HumanMessage(content=prompt)])
            answer_text = response["choices"][0]["message"]["content"]

        except Exception as e:
            print(f"⚠ ClaraAgent LLM Error: {e}")
            import traceback
            traceback.print_exc()
            answer_text = "Unable to generate an answer with the current data."

        payload["answer"] = {"text": answer_text.strip(), "confidence": 0.7}
        payload["finalized"] = True

        print("\n📝 Debug - Docs passed to ClaraAgent:")
        print(context)
        return payload