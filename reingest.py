# reingest.py
import os
from src.vector_store import FlightVectorStore

# DEFAULTS
DEFAULT_CSV = os.path.join("data", "flight_data.csv")  # default CSV file path
DEFAULT_COLLECTION = "default"
DEFAULT_BATCH_SIZE = 500

def main():
    # Pick CSV file
    csv_path = DEFAULT_CSV
    collection_name = DEFAULT_COLLECTION
    batch_size = DEFAULT_BATCH_SIZE

    print(f"📂 Loading CSV: {csv_path}")
    print(f"🗃 Using collection: {collection_name}")
    print(f"⚡ Batch size: {batch_size}")

    # Initialize vector store
    store = FlightVectorStore(collection_name=collection_name)

    # Ingest CSV into vector DB
    store.ingest_csv(csv_path=csv_path, batch_size=batch_size)

    print("✅ Reingestion complete!")


if __name__ == "__main__":
    main()
