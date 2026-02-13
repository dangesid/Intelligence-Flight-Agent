import os
from src.vector_store import FlightVectorStore

# Cosmos ENV variables
COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT")
COSMOS_KEY = os.getenv("COSMOS_KEY")
COSMOS_DATABASE = os.getenv("COSMOS_DATABASE")
COSMOS_CONTAINER = os.getenv("COSMOS_CONTAINER")

DEFAULT_COLLECTION = "default"
DEFAULT_BATCH_SIZE = 500


def main():
    print("🚀 Starting full Cosmos → Chroma reingestion")

    store = FlightVectorStore(
        collection_name=DEFAULT_COLLECTION
    )

    store.ingest_cosmos(
        endpoint=COSMOS_ENDPOINT,
        key=COSMOS_KEY,
        database_name=COSMOS_DATABASE,
        container_name=COSMOS_CONTAINER,
        batch_size=DEFAULT_BATCH_SIZE,
    )

    print("✅ Reingestion complete!")


if __name__ == "__main__":
    main()
