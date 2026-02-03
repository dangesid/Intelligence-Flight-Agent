# Flight Knowledge Vector Store

A lightweight utility for ingesting flight CSV data, generating embeddings with SentenceTransformers or Ollama, and storing/querying them in Chroma (duckdb+parquet). Designed as a vector store backend for flight-focused GenAI assistants with intelligent query processing and knowledge gap analysis.

---

## Features

- **Ingest flight CSVs** and build human-readable flight descriptions per row
- **Compute sentence embeddings** via SentenceTransformer models or Ollama
- **Store vectors, metadata, and documents** in Chroma (duckdb+parquet backend)
- **Intelligent query processing** with intent recognition and routing
- **Knowledge gap detection** to identify missing information
- **CLARA reasoning system** for complex query handling
- **Simple query API** to retrieve top-k relevant flight documents with distances and metadata

---

## Repository Structure

```
genai_knowledge_bot/
├── src/
│   ├── agents/
│   │   ├── query_intent_agent.py    # Query intent classification
│   │   └── knowledge_gap_agent.py   # Knowledge gap detection
│   ├── evaluation/
│   │   └── gap_analyzer.py          # Analyze and report knowledge gaps
│   ├── ingestion/
│   │   └── vector_store.py          # FlightVectorStore class: ingest, query, inspect
│   ├── llm_engine/
│   │   ├── base.py                  # Base LLM interface
│   │   ├── factory.py               # LLM factory pattern
│   │   └── ollama_client.py         # Ollama client implementation
│   ├── reasoning/
│   │   └── clara_system.py          # CLARA reasoning system
│   ├── config.py                    # App settings and configuration
│   └── main.py                      # CLI entrypoint
├── data/                            # Sample flight CSV files
├── vector_db/                       # Chroma database storage
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── project_structure.py             # Project structure utility
```

---

## Prerequisites

- **macOS** with Python 3.8+ (tested with Python 3.11)
- **pip** and **virtualenv/venv**
- **Ollama** (if using Ollama for embeddings/LLM)
- Optional: GPU for faster inference

---

## Quick Setup

### 1. Create and activate a virtual environment

```bash
python3 -m venv kbBot
source kbBot/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install and setup Ollama

**Download and install Ollama:**

```bash
# Install Ollama on macOS
curl -fsSL https://ollama.com/install.sh | sh
```

Or download from [https://ollama.com/download](https://ollama.com/download)

**Pull required models:**

```bash
# Pull embedding model (required for vector search)
ollama pull nomic-embed-text

# Pull LLM model (required for query processing)
ollama pull llama3.2

# Alternative models:
# For embeddings:
ollama pull mxbai-embed-large

# For LLM:
ollama pull mistral
ollama pull llama3.1
```

**Verify Ollama is running:**

```bash
ollama list
```

**Start Ollama service (if not running):**

```bash
ollama serve
```

### 4. Configure environment variables

Create a `.env` file in the project root or set environment variables:

```bash
# Vector database configuration
export VECTOR_DB_PATH="./vector_db"

# Embedding model configuration
export EMBEDDING_MODEL="nomic-embed-text"
export USE_OLLAMA="true"

# LLM configuration
export OLLAMA_MODEL="llama3.2"
export OLLAMA_BASE_URL="http://localhost:11434"

# Optional: Temperature and other LLM parameters
export LLM_TEMPERATURE="0.7"
```

---

## CSV Format

The ingestion script expects CSV files with the following columns:

- `flight_number`
- `origin`
- `destination`
- `departure_time`
- `arrival_time`
- `airline`
- `price`

**Example CSV (`data/flights.csv`):**

```csv
flight_number,origin,destination,departure_time,arrival_time,airline,price
AA123,SFO,JFK,2026-02-15T08:00,2026-02-15T16:30,American Airlines,299
UA456,LAX,ORD,2026-02-16T09:00,2026-02-16T15:00,United Airlines,199
DL789,ATL,MIA,2026-02-17T10:00,2026-02-17T12:30,Delta Airlines,149
```

---

## Usage

### Running Ingestion

```bash
# Ingest flight data from CSV
python -m src.main ingest data/flights.csv
```

### Querying the Vector Store

```bash
# Simple query
python -m src.main query "flights from SFO to JFK under $300"

# Complex query with reasoning
python -m src.main query "What are the cheapest morning flights from LAX?"
```

### Inspecting the Database

```bash
# View database statistics
python -m src.main inspect
```

### Knowledge Gap Analysis

```bash
# Analyze what information is missing
python -m src.main analyze-gaps
```

---

## Ollama Model Recommendations

### For Embeddings:
- **`nomic-embed-text`** - Lightweight, fast, good quality (recommended)
- **`mxbai-embed-large`** - Higher quality, larger model
- **`all-minilm`** - Compact alternative

### For LLM (Query Processing):
- **`llama3.2`** - Fast, efficient, good reasoning (recommended)
- **`llama3.1`** - More capable, larger context
- **`mistral`** - Good balance of speed and quality
- **`mixtral`** - High quality, slower

### To switch models:

```bash
# Pull new model
ollama pull mistral

# Update environment variable
export OLLAMA_MODEL="mistral"

# Restart your application
python -m src.main query "your query here"
```

---

## Architecture Overview

### Components:

1. **Vector Store** (`ingestion/vector_store.py`)
   - Handles CSV ingestion and embedding generation
   - Manages Chroma database operations

2. **LLM Engine** (`llm_engine/`)
   - Factory pattern for different LLM providers
   - Ollama client for local inference

3. **Agents** (`agents/`)
   - Query intent classification
   - Knowledge gap detection

4. **CLARA System** (`reasoning/clara_system.py`)
   - Advanced reasoning for complex queries
   - Multi-step query decomposition

5. **Evaluation** (`evaluation/gap_analyzer.py`)
   - Identifies missing data
   - Suggests improvements

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **Ollama connection error** | Ensure Ollama is running: `ollama serve` |
| **Model not found** | Pull the model: `ollama pull <model-name>` |
| **Port already in use** | Check if Ollama is running on port 11434: `lsof -i :11434` |
| **Slow embedding generation** | Use a smaller model like `nomic-embed-text` |
| **Out of memory** | Use a smaller LLM model or reduce context size |
| **Chroma database errors** | Delete `vector_db/` and re-ingest data |
| **CSV parsing errors** | Verify CSV format matches expected columns |

---

## Development

### Running tests:

```bash
pytest tests/
```

### Project structure:

```bash
python project_structure.py
```

---

## Contributing

1. Fork the repository
2. Create a feature branch from `main`
3. Implement tests for new behavior
4. Open a PR with a clear description

Keep changes minimal and include migration notes if dependencies change.

---

## License

This project is licensed under the MIT License.

---

## Additional Resources

- [Ollama Documentation](https://ollama.com/docs)
- [ChromaDB Documentation](https://docs.trychroma.com)
- [SentenceTransformers Documentation](https://www.sbert.net)

---