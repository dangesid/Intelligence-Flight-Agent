from typing import List
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from src.config import settings


class FlightVectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=settings.VECTOR_DB_PATH
        )

        self.collection = self.client.get_or_create_collection(
            name="flights"
        )

        self.embedder = SentenceTransformer(settings.EMBEDDING_MODEL)

    def ingest_csv(self, csv_path: str, batch_size: int = 500):
        df = pd.read_csv(csv_path)

        # Normalize columns
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        print("📌 Detected columns:", df.columns.tolist())

        def build_text(row):
            parts = []

            if "flight" in row:
                parts.append(f"Flight {row['flight']}")

            if "origin" in row and "dest" in row:
                parts.append(f"from {row['origin']} to {row['dest']}")

            if "dep_time" in row:
                parts.append(f"departs at {row['dep_time']}")

            if "arr_time" in row:
                parts.append(f"arrives at {row['arr_time']}")

            if "carrier" in row:
                parts.append(f"carrier {row['carrier']}")

            if "distance" in row:
                parts.append(f"distance {row['distance']} miles")

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
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
        )
        return results["documents"][0]
    
    def query_with_scores(self, query_text: str, n_results: int = 5) -> List[tuple]:
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

        metadatas = results["metadatas"]

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

