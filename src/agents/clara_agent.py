# src/agents/clara_agent.py
from typing import Dict, Any
from langchain_core.messages import HumanMessage
from src.llm_clients.azure_openai_client import AzureOpenAIWrapper


class ClaraAgent:
    def __init__(self):
        # Use your custom wrapper instead of LangChain's AzureChatOpenAI
        self.llm = AzureOpenAIWrapper()

    def can_handle(self, payload: Dict[str, Any]) -> bool:
        """
        ClaraAgent handles the query when:
        1. Context has been retrieved (from RAG or other sources)
        2. The query hasn't been finalized yet
        """
        has_context = bool(payload.get("context"))
        not_finalized = not payload.get("finalized", False)
        
        return has_context and not_finalized

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        query = payload["query"]
        context = payload.get("context", "")

        if not context:
            payload["answer"] = {"text": "Data not available.", "confidence": 0.0}
            payload["finalized"] = True
            return payload

        try:
            # Build prompt with context
            prompt = (
                "You are a helpful flight assistant. "
                "Answer strictly using the provided context. "
                "If the information is not present, say 'Data not available'. "
                "Do NOT use any external knowledge or make assumptions.\n\n"
                f"Context:\n{context}\n\nQuestion:\n{query}"
            )

            # Use your custom wrapper's generate method
            response = self.llm.generate(messages=[HumanMessage(content=prompt)])
            
            # Extract text from the Azure response
            # Azure returns: {"choices": [{"message": {"content": "..."}}]}
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