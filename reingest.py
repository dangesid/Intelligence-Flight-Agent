#!/usr/bin/env python3
"""
Re-ingest flights data into vector store with enhanced document text.
"""
from src.ingestion.vector_store import FlightVectorStore

print("\n🗑️  Clearing and re-ingesting vector database...")
vs = FlightVectorStore()
vs.ingest_csv('data/flights.csv')
print("\n✅ Complete! Vector DB now has airline names for better retrieval.")
