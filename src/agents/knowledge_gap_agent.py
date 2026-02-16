# src/agents/knowledge_gap_agent.py

from typing import Dict, Any
from src.agents.base_agent import BaseAgent
from src.llm_clients.azure_openai_client import AzureOpenAIWrapper
from langchain_core.messages import HumanMessage
import json


class KnowledgeGapAgent(BaseAgent):
    """
    Detects knowledge gaps by validating if retrieved context
    is relevant to user's query using Azure GPT.
    """

    def __init__(self):
        self.llm = AzureOpenAIWrapper()

    def can_handle(self, payload: Dict[str, Any]) -> bool:
        # 🔥 IMPORTANT: match orchestrator flag
        return (
            "query" in payload
            and "context" in payload
            and not payload.get("gap_checked", False)
        )

    def run(self, payload: dict) -> dict:

        query = payload.get("query", "")
        context = payload.get("context", "")

        # 🔥 CRITICAL: Prevent infinite loop
        payload["gap_checked"] = True

        # ---------------------------------------------------
        # CASE 1: No Context Retrieved
        # ---------------------------------------------------
        if not context:
            payload["gap_signal"] = {
                "reason": "No relevant documents retrieved.",
                "severity": "medium",
                "llm_analysis": {
                    "knowledge_gap": "No documents were retrieved from the knowledge base.",
                    "recommended_data_to_add": []
                }
            }
            return payload

        # ---------------------------------------------------
        # CASE 2: Validate via LLM
        # ---------------------------------------------------
        try:
            prompt = f"""
You are a strict JSON validator.

User Query:
{query}

Retrieved Context:
{context}

Return ONLY valid JSON:
{{
  "relevant": true or false,
  "answerable": true or false,
  "reason": "...",
  "severity": "low|medium|high",
  "knowledge_gap": "...",
  "recommended_data_to_add": ["item1", "item2"]
}}
"""

            response = self.llm.generate(
                messages=[HumanMessage(content=prompt)]
            )

            json_text = response["choices"][0]["message"]["content"]
            result = json.loads(json_text)

            # If not relevant OR not answerable → GAP
            if not result.get("relevant", True) or not result.get("answerable", True):

                payload["gap_signal"] = {
                    "reason": result.get("reason", "Unknown reason"),
                    "severity": result.get("severity", "medium"),
                    "llm_analysis": {
                        "knowledge_gap": result.get("knowledge_gap", ""),
                        "recommended_data_to_add": result.get(
                            "recommended_data_to_add", []
                        )
                    }
                }

            else:
                payload["gap_signal"] = None

        except Exception as e:
            print(f"⚠️ KnowledgeGapAgent LLM Error: {e}")

            payload["gap_signal"] = {
                "reason": "LLM failed to validate relevance.",
                "severity": "medium",
                "llm_analysis": {
                    "knowledge_gap": "LLM validation failure.",
                    "recommended_data_to_add": []
                }
            }

        return payload
