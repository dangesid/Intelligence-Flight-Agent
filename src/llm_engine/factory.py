# src/llm_engine/factory.py
import os


from langchain_openai import AzureChatOpenAI



def get_llm():
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        model=os.getenv("AZURE_OPENAI_MODEL", "gpt-35-turbo"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0.0,
        max_tokens=1000,
    )
    return llm