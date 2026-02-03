import os

PROJECT_STRUCTURE = {
    "genai_knowledge_bot": {
        "data": {
            "flights.csv": ""
        },
        "src": {
            "__init__.py": "",
            "config.py": "",
            "llm_engine": {
                "__init__.py": "",
                "base.py": "",
                "ollama_client.py": "",
                "factory.py": ""
            },
            "ingestion": {
                "__init__.py": "",
                "vector_store.py": ""
            },
            "reasoning": {
                "__init__.py": "",
                "clara_system.py": ""
            },
            "evaluation": {
                "__init__.py": "",
                "gap_analyzer.py": ""
            },
            "main.py": ""
        },
        "requirements.txt": "",
        ".env": ""
    }
}


def create_structure(base_path, structure):
    for name, content in structure.items():
        path = os.path.join(base_path, name)

        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(content)


if __name__ == "__main__":
    create_structure(".", PROJECT_STRUCTURE)
    print("✅ genai_knowledge_bot project structure created successfully!")
