from typing import List
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from src.config import settings


class FlightVectorStore:
    """
    Vector store for flight data using ChromaDB and sentence embeddings.

    - Dynamically ingests all columns from CSV
    - Generates embeddings for the entire row content
    - Allows semantic queries with retrieval of documents, distances, and metadata
    """

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=settings.VECTOR_DB_PATH
        )

        self.collection = self.client.get_or_create_collection(
            name="flights"
        )

        self.embedder = SentenceTransformer(settings.EMBEDDING_MODEL)

    def ingest_csv(self, csv_path: str, batch_size: int = 500):
        """
        Ingest all rows and all columns from a CSV into Chroma vector DB
        """
        df = pd.read_csv(csv_path)

        # Normalize column names
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        print("📌 Detected columns:", df.columns.tolist())

        # Convert each row into a single text string using all columns
        def build_text(row):
            parts = []
            for col, val in row.items():
                if pd.notna(val):
                    parts.append(f"{col}: {val}")
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

        print("🎉 All flights successfully ingested into Chroma DB")

    def query(self, query_text: str, n_results: int = 5) -> List[str]:
        """
        Simple query for top documents without distances
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

    def inspect_domain(self, limit: int = 100):
        """
        Inspect what data actually exists in the Vector DB
        """
        results = self.collection.get(
            include=["metadatas"],
            limit=limit
        )

        metadatas = results.get("metadatas", [])

        origins = {m.get("origin") for m in metadatas if m.get("origin")}
        destinations = {m.get("dest") for m in metadatas if m.get("dest")}
        carriers = {m.get("carrier") for m in metadatas if m.get("carrier")}
        flights = {m.get("flight") for m in metadatas if m.get("flight")}

        return {
            "origins": sorted(origins),
            "destinations": sorted(destinations),
            "carriers": sorted(carriers),
            "flights": sorted(flights)
        }

    def lookup_by_flight_id(self, flight_number: str):
        """
        Exact flight lookup using Chroma metadata (not embeddings)
        """
        results = self.collection.get(
            where={"flight": int(flight_number)},
            include=["metadatas"]
        )

        metadatas = results.get("metadatas", [])

        if not metadatas:
            return None

        return metadatas[0]
