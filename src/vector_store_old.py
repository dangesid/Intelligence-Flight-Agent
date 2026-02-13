from typing import List, Dict, Any
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from src.config_ollama import settings


class FlightVectorStore:
    """
    General-purpose vector store using ChromaDB and sentence embeddings.

    Features:
    - Dynamically ingests all columns from any CSV
    - Generates embeddings for entire row content
    - Allows semantic queries with retrieval of documents, distances, and metadata
    """

    def __init__(self, collection_name: str = "default"):
        self.client = chromadb.PersistentClient(
            path=settings.VECTOR_DB_PATH
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

        self.embedder = SentenceTransformer(settings.EMBEDDING_MODEL)

    def ingest_csv(self, csv_path: str, batch_size: int = 500):
        """
        Ingest all rows and columns from a CSV into Chroma vector DB
        """
        df = pd.read_csv(csv_path)

        # Normalize column names
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        print("📌 Detected columns:", df.columns.tolist())

        # Convert each row into a single text string using all columns
        def build_text(row):
            parts = [f"{col}: {val}" for col, val in row.items() if pd.notna(val)]
            return ", ".join(parts)

        df["text"] = df.apply(build_text, axis=1)
        texts = df["text"].tolist()
        metadatas = df.to_dict(orient="records")

        print(f"🚀 Ingesting {len(texts)} documents into Chroma in batches...")

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_meta = metadatas[i:i + batch_size]

            embeddings = self.embedder.encode(
                batch_texts,
                show_progress_bar=False
            )

            self.collection.add(
                documents=batch_texts,
                metadatas=batch_meta,
                ids=[str(i + j) for j in range(len(batch_texts))],
                embeddings=embeddings.tolist(),
            )

            print(f"✅ Ingested batch {i // batch_size + 1}")

        print("🎉 All data successfully ingested into Chroma DB")

    def query(self, query_text: str, n_results: int = 5) -> List[str]:
        """
        Semantic query for top documents
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
        )
        return results["documents"][0]

    def query_with_scores(self, query_text: str, n_results: int = 5) -> List[tuple]:
        """
        Query top documents with distances and metadata
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=["documents", "distances", "metadatas"],
        )
        documents = results["documents"][0]
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]
        return list(zip(documents, distances, metadatas))

    def inspect_metadata(self, limit: int = 100) -> Dict[str, set]:
        """
        Inspect metadata keys and their unique values dynamically
        """
        results = self.collection.get(
            include=["metadatas"],
            limit=limit
        )
        metadatas = results.get("metadatas", [])

        if not metadatas:
            return {}

        # Collect unique values for all metadata keys
        keys = metadatas[0].keys()
        unique_values = {k: set() for k in keys}
        for meta in metadatas:
            for k, v in meta.items():
                if v is not None:
                    unique_values[k].add(v)

        # Sort the sets for readability
        return {k: sorted(v) for k, v in unique_values.items()}

    def lookup_by_metadata(self, key: str, value: Any):
        """
        Generic metadata-based lookup
        """
        results = self.collection.get(
            where={key: value},
            include=["metadatas"]
        )
        metadatas = results.get("metadatas", [])
        return metadatas if metadatas else None
