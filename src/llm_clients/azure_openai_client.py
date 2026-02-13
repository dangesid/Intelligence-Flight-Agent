# src/llm_clients/azure_openai_client.py
import os
import requests
from langchain_core.messages import HumanMessage


class AzureOpenAIWrapper:
    def __init__(self):
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        self.api_version = "2025-01-01-preview"

        if not all([self.endpoint, self.api_key, self.deployment]):
            raise ValueError(
                "Missing Azure OpenAI environment variables."
            )

    def generate(self, messages: list[HumanMessage]) -> dict:
        """
        messages: list of HumanMessage objects
        Returns dict containing the Azure GPT response
        """
        # IMPORTANT: Extract content from HumanMessage objects
        azure_messages = []
        for m in messages:
            if isinstance(m, HumanMessage):
                azure_messages.append({"role": "user", "content": m.content})
            elif isinstance(m, str):
                # Handle string accidentally passed
                azure_messages.append({"role": "user", "content": m})
            else:
                # Handle dict or other types
                azure_messages.append({"role": "user", "content": str(m)})

        url = f"{self.endpoint.rstrip('/')}/openai/deployments/{self.deployment}/chat/completions?api-version={self.api_version}"

        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }

        data = {"messages": azure_messages}

        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Azure API Error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")
            raise