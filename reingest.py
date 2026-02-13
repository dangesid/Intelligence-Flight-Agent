import os
import json
import traceback
from dotenv import load_dotenv
from src.vector_store import FlightVectorStore
from src.config import settings

load_dotenv()

STATE_FILE = "system_state.json"

COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT")
COSMOS_KEY = os.getenv("COSMOS_KEY")
COSMOS_DATABASE = os.getenv("COSMOS_DATABASE")
COSMOS_CONTAINER = os.getenv("COSMOS_CONTAINER")

DEFAULT_COLLECTION = "default"
DEFAULT_BATCH_SIZE = 500


# ==================================================
# STATE MANAGEMENT
# ==================================================

def save_active_source(source: str):
    state = {
        "active_vector_source": source
    }

    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)

    print(f"📌 System now set to use: {source}")


# ==================================================
# COSMOS INGESTION (PRIMARY)
# ==================================================

def ingest_cosmos_primary():
    print("\n🚀 Attempting Cosmos ingestion (PRIMARY)...\n")

    store = FlightVectorStore(
        collection_name=DEFAULT_COLLECTION,
        vector_db_path=settings.VECTOR_DB_PATH_COSMOS,
        reset_collection=True,  # important
    )

    store.ingest_cosmos(
        endpoint=COSMOS_ENDPOINT,
        key=COSMOS_KEY,
        database_name=COSMOS_DATABASE,
        container_name=COSMOS_CONTAINER,
        batch_size=DEFAULT_BATCH_SIZE,
    )

    save_active_source("COSMOS")
    print("\n✅ Cosmos ingestion successful.\n")


# ==================================================
# CSV FALLBACK
# ==================================================

def ingest_csv_fallback():
    print("\n⚠️ Switching to CSV fallback ingestion...\n")

    CSV_FILE_PATH = os.getenv("CSV_FILE_PATH")

    if not CSV_FILE_PATH:
        raise Exception("CSV_FILE_PATH not set in .env")

    store = FlightVectorStore(
        collection_name=DEFAULT_COLLECTION,
        vector_db_path=settings.VECTOR_DB_PATH_CSV,
        reset_collection=True,
    )

    store.ingest_csv(
        csv_path=CSV_FILE_PATH,
        batch_size=DEFAULT_BATCH_SIZE,
    )

    save_active_source("CSV")
    print("\n✅ CSV ingestion successful.\n")


# ==================================================
# MAIN LOGIC
# ==================================================

def main():
    try:
        ingest_cosmos_primary()

    except Exception as e:
        print("\n❌ COSMOS INGESTION FAILED\n")
        print("Reason:")
        print(str(e))
        print("\nFull Traceback:")
        traceback.print_exc()

        ingest_csv_fallback()


if __name__ == "__main__":
    main()
