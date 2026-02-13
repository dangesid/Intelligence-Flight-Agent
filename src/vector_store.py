from typing import List, Any
from sentence_transformers import SentenceTransformer
import chromadb
from azure.cosmos import CosmosClient
from src.config import settings
import os


class FlightVectorStore:
    """
    Production-ready vector store.

    Architecture:
        Cosmos DB (or CSV) → Embeddings → Chroma Persistent Store

    Features:
    - Smart DB detection (Cosmos / CSV)
    - Clean reingestion when requested
    - Batch ingestion
    - Persistent storage
    - Query with scores
    """

    def __init__(
        self,
        collection_name: str = "default",
        vector_db_path: str = None,
        reset_collection: bool = False,  # important improvement
    ):
        # Determine active vector DB path
        db_path = vector_db_path or settings.VECTOR_DB_PATH_COSMOS

        self.vector_db_path = db_path
        self.collection_name = collection_name

        # Ensure folder exists
        os.makedirs(db_path, exist_ok=True)

        # Log which DB path is active
        if "cosmos" in db_path.lower():
            print("🌌 VECTOR STORE MODE: COSMOS (Primary)")
        elif "csv" in db_path.lower():
            print("📄 VECTOR STORE MODE: CSV (Fallback)")
        else:
            print(f"📁 VECTOR STORE MODE: Custom Path ({db_path})")

        print(f"📁 Using vector DB path: {db_path}")

        self.client = chromadb.PersistentClient(path=db_path)

        existing_collections = [
            col.name for col in self.client.list_collections()
        ]

        # Only delete if explicitly requested
        if reset_collection and collection_name in existing_collections:
            print("🧹 Resetting collection for clean reingestion...")
            self.client.delete_collection(collection_name)

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

        # Load embedding model
        print("🧠 Loading embedding model...")
        self.embedder = SentenceTransformer(settings.EMBEDDING_MODEL)

    # ==================================================
    # COSMOS INGESTION
    # ==================================================

    def ingest_cosmos(
        self,
        endpoint: str,
        key: str,
        database_name: str,
        container_name: str,
        batch_size: int = 500,
    ):
        print("🔌 Connecting to Azure Cosmos DB...")

        try:
            client = CosmosClient(endpoint, key)
            database = client.get_database_client(database_name)
            container = database.get_container_client(container_name)
        except Exception as e:
            print("❌ Cosmos connection failed.")
            raise Exception(f"Cosmos connection error: {str(e)}")

        print("📥 Fetching documents from Cosmos DB...")

        query = "SELECT * FROM c"

        try:
            items = container.query_items(
                query=query,
                enable_cross_partition_query=True,
            )
        except Exception as e:
            print("❌ Cosmos query failed.")
            raise Exception(f"Cosmos query error: {str(e)}")

        texts = []
        metadatas = []
        ids = []

        total = 0
        batch_count = 0

        for item in items:
            clean_item = {
                k: v for k, v in item.items()
                if not k.startswith("_")
            }

            parts = [
                f"{k}: {v}"
                for k, v in clean_item.items()
                if v is not None
            ]

            text = ", ".join(parts)

            texts.append(text)
            metadatas.append(clean_item)

            doc_id = str(clean_item.get("id", total))
            ids.append(doc_id)

            if len(texts) >= batch_size:
                self._add_batch(texts, metadatas, ids)
                total += len(texts)
                batch_count += 1
                print(f"✅ Ingested batch {batch_count} ({total} docs)")
                texts, metadatas, ids = [], [], []

        if texts:
            self._add_batch(texts, metadatas, ids)
            total += len(texts)

        print(f"🎉 Cosmos ingestion complete ({total} documents)")


        # ==================================================
    # CSV INGESTION (Fallback)
    # ==================================================

    def ingest_csv(
        self,
        csv_path: str,
        batch_size: int = 500,
    ):
        import csv

        print(f"📄 Loading CSV from: {csv_path}")

        texts = []
        metadatas = []
        ids = []

        total = 0
        batch_count = 0

        with open(csv_path, newline='', encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                clean_row = {k: v for k, v in row.items() if v}

                parts = [
                    f"{k}: {v}"
                    for k, v in clean_row.items()
                ]

                text = ", ".join(parts)

                texts.append(text)
                metadatas.append(clean_row)

                doc_id = str(clean_row.get("id", total))
                ids.append(doc_id)

                if len(texts) >= batch_size:
                    self._add_batch(texts, metadatas, ids)
                    total += len(texts)
                    batch_count += 1
                    print(f"✅ CSV batch {batch_count} ({total} docs)")
                    texts, metadatas, ids = [], [], []

        if texts:
            self._add_batch(texts, metadatas, ids)
            total += len(texts)

        print(f"🎉 CSV ingestion complete ({total} documents)")


    # ==================================================
    # INTERNAL BATCH ADD
    # ==================================================

    def _add_batch(self, texts, metadatas, ids):
        embeddings = self.embedder.encode(
            texts,
            show_progress_bar=False,
        )

        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings.tolist(),
        )

    # ==================================================
    # QUERY METHODS
    # ==================================================

    def query(self, query_text: str, n_results: int = 5) -> List[str]:
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
        )

        return results["documents"][0] if results["documents"] else []

    def query_with_scores(self, query_text: str, n_results: int = 5):
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=["documents", "distances", "metadatas"],
        )

        if not results["documents"]:
            return []

        documents = results["documents"][0]
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]

        return list(zip(documents, distances, metadatas))

    def inspect_metadata(self, limit: int = 100):
        results = self.collection.get(
            include=["metadatas"],
            limit=limit,
        )

        metadatas = results.get("metadatas", [])
        if not metadatas:
            return {}

        keys = metadatas[0].keys()
        unique_values = {k: set() for k in keys}

        for meta in metadatas:
            for k, v in meta.items():
                if v is not None:
                    unique_values[k].add(v)

        return {k: sorted(v) for k, v in unique_values.items()}

    def lookup_by_metadata(self, key: str, value: Any):
        results = self.collection.get(
            where={key: value},
            include=["metadatas"],
        )

        metadatas = results.get("metadatas", [])
        return metadatas if metadatas else None
