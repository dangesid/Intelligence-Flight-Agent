# src/agents/knowledge_gap_agent.py
from typing import Dict, Any
from src.agents.base_agent import BaseAgent
from src.llm_clients.azure_openai_client import AzureOpenAIWrapper
from langchain_core.messages import HumanMessage
import json
import re


class KnowledgeGapAgent(BaseAgent):
    """
    Detects knowledge gaps by validating if retrieved context
    is relevant to user's query using Azure GPT.
    """

    def __init__(self):
        # Create its own LLM instance
        self.llm = AzureOpenAIWrapper()

    def can_handle(self, payload: Dict[str, Any]) -> bool:
        return "query" in payload and "context" in payload and not payload.get("_knowledge_gap_done", False)

    def run(self, payload: dict) -> dict:
        query = payload.get("query", "")
        context = payload.get("context", "")

        if not context:
            payload["gap_signal"] = {
                "reason": "No relevant documents retrieved.",
                "severity": "medium"
            }
            payload["_knowledge_gap_done"] = True
            return payload

        # Run LLM only if context exists
        try:
            prompt = f"""
You are a strict JSON validator.
User Query:
{query}

Retrieved Context:
{context}

Return JSON:
{{"relevant": true or false, "answerable": true or false, "reason": "...", "severity": "low|medium|high", "missing_knowledge": "..."}}
"""
            # Use the wrapper's generate method correctly
            response = self.llm.generate(messages=[HumanMessage(content=prompt)])
            
            # Extract JSON from Azure response
            json_text = response["choices"][0]["message"]["content"]
            result = json.loads(json_text)
            
            payload["llm_analysis"] = result

            if not result.get("relevant", True) or not result.get("answerable", True):
                payload["gap_signal"] = {
                    "reason": result.get("reason", "Unknown"),
                    "severity": result.get("severity", "medium")
                }
            else:
                payload["gap_signal"] = None

        except Exception as e:
            print(f"⚠️ KnowledgeGapAgent LLM Error: {e}")
            import traceback
            traceback.print_exc()
            
            payload["gap_signal"] = {
                "reason": "LLM failed to validate relevance.",
                "severity": "medium"
            }

        payload["_knowledge_gap_done"] = True
        return payload