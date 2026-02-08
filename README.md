# Flight Knowledge Vector Store

A lightweight utility for ingesting tabular data (CSV/TSV/structured formats), generating embeddings with SentenceTransformers or Ollama, and storing/querying them in Chroma (duckdb+parquet). This repository provides ingestion, semantic search, intent routing, knowledge-gap detection, and a CLARA reasoning system for multi-step queries.

---

## What this repo does (summary)

- Ingests structured tabular data (CSV) and converts rows into human-readable documents.
- Creates embeddings for each document using SentenceTransformers or Ollama embedding models.
- Stores documents, metadata, and vectors in ChromaDB (persistent client, duckdb+parquet).
- Exposes CLI entrypoints (`src/main.py`) for ingest, query, inspect, and analyze-gaps.
- Routes queries through lightweight agents (intent detection, retrieval, gap analysis) and CLARA for complex reasoning.

---

## Quick Start — minimal steps to run the repo

1. Create & activate Python environment

```bash
python3 -m venv kbBot
source kbBot/bin/activate
```

2. Install Python dependencies

```bash
pip install -r requirements.txt
```

3. (If using Ollama) Install and run Ollama, then pull models

```bash
# Install Ollama (macOS)
curl -fsSL https://ollama.com/install.sh | sh

# Start the Ollama service (background)
ollama serve &

# Pull embedding model (recommended)
ollama pull nomic-embed-text

# Pull LLM for query processing (recommended)
ollama pull llama3.2
```

4. Configure environment variables (create `.env` or export)

Edit `.env` or export variables used by `src/config.py`:

- `LLM_PROVIDER` (default `ollama`) — provider selection
- `OLLAMA_MODEL` — model name used by Ollama (e.g. `llama3.2`)
- `OLLAMA_BASE_URL` — `http://localhost:11434` by default
- `VECTOR_DB_PATH` — where Chroma stores DB (default `./vector_db`)
- `EMBEDDING_MODEL` — SentenceTransformers model name (if not using Ollama for embeddings)

Example `.env`:

```
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434
VECTOR_DB_PATH=./vector_db
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

5. Provide your CSV (or change path)

- Default ingestion path: `data/flights.csv`. To change, edit `reingest.py` (variable `DEFAULT_CSV`) or call the ingestion CLI with a path.
- Ingestion code reads CSV via `src/vector_store.py` (`ingest_csv` uses `pd.read_csv`).

6. Ingest data (build vectors)

```bash
python reingest.py
```

7. Run queries / inspect

```bash
# Query examples using the CLI
python -m src.main query "flights from SFO to JFK under $300"
python -m src.main inspect
python -m src.main analyze-gaps
```

---

## Repository Structure

```
genai_knowledge_bot/
├── src/
│   ├── agents/
│   │   ├── base_agent.py            # Base agent class
│   │   ├── query_intent_agent.py    # Query intent classification
│   │   ├── retrieval_agent.py       # Semantic retrieval from vector store
│   │   ├── knowledge_gap_agent.py   # Knowledge gap detection
│   │   ├── context_builder_agent.py # Builds LLM prompts from docs
│   │   ├── final_answer_agent.py    # Formats final response
│   │   ├── vector_agent.py          # Vector store operations helper
│   │   └── clara_agent.py           # CLARA reasoning agent
│   ├── evaluation/
│   │   └── gap_analyzer.py          # Analyze and report knowledge gaps
│   ├── ingestion/
│   │   └── (migration note: logic moved to src/vector_store.py)
│   ├── llm_engine/
│   │   ├── base.py                  # Base LLM interface
│   │   ├── factory.py               # LLM factory pattern (provider selection)
│   │   └── ollama_client.py         # Ollama client implementation
│   ├── orchestrator/
│   │   └── orchestrator.py          # Query pipeline orchestration
│   ├── reasoning/
│   │   └── clara_agent.py           # CLARA reasoning system
│   ├── config.py                    # App settings and configuration
│   ├── vector_store.py              # Vector store: ingest, query, inspect
│   ├── main.py                      # CLI entrypoint
│   └── main_old.py                  # (legacy)
├── data/
│   └── flights.csv                  # Sample data (CSV ingestion format)
├── vector_db/                       # Chroma database storage (created on ingest)
├── reingest.py                      # Ingestion script (main entry for CSV → DB)
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── project_structure.py             # Project structure utility
```

---

## Generic Data Ingestion — How to use with any CSV

The ingestion pipeline is **generic** and works with any tabular CSV format:

### Step-by-step:

1. **Prepare your CSV** with a header row. Column names can be any text; the ingestion code normalizes them (lowercase, replace spaces with underscores).

2. **Call the ingestion**:

   - **CLI**: Update `reingest.py` (`DEFAULT_CSV` = path to your CSV), then run:
   
     ```bash
     python reingest.py
     ```
   
   - **Programmatically**: 
   
     ```python
     from src.vector_store import FlightVectorStore
     
     store = FlightVectorStore(collection_name="my_data")
     store.ingest_csv("path/to/your/data.csv", batch_size=500)
     ```

3. **How ingestion works** (see `src/vector_store.py` `ingest_csv` method):
   
   - Each row is converted into a single text string: `"col1: value1, col2: value2, ..."`
   - The text string is embedded using the configured embedding model (SentenceTransformer or Ollama).
   - Document (text), metadata (entire row as dict), and embedding vector are stored in ChromaDB.
   - Null/empty cells are skipped automatically.

4. **Tips for best results**:
   
   - Use consistent, descriptive column names.
   - Avoid extremely long string values per cell (they dilute embeddings).
   - Dates: use ISO format (e.g., `2026-02-15T08:00`) for semantic clarity.
   - Numeric fields: can be included as-is; embeddings will recognize numeric patterns.

---

## Pulling Ollama Models and Using Them

Ollama is a local LLM/embedding server. To use it:

### Install Ollama

```bash
# macOS
curl -fsSL https://ollama.com/install.sh | sh

# Or download from https://ollama.com/download
```

### Start Ollama server

```bash
ollama serve
# Runs on http://localhost:11434 by default
```

### Pull models

```bash
# For embeddings (lightweight, fast)
ollama pull nomic-embed-text

# For LLM (query processing / reasoning)
ollama pull llama3.2
ollama pull mistral
ollama pull llama3.1

# List pulled models
ollama list
```

### Use in this project

- Set `LLM_PROVIDER=ollama` in `.env` or `src/config.py`.
- Set `OLLAMA_MODEL` to your chosen model (e.g., `llama3.2`).
- Configure `OLLAMA_BASE_URL` (default: `http://localhost:11434`).
- The project uses the configured model via `src/llm_engine/ollama_client.py`.

### Switch models at runtime

```bash
# Pull a new model
ollama pull mistral

# Update .env
export OLLAMA_MODEL=mistral

# Re-run your query
python -m src.main query "your question"
```

---

## Agents, CLARA, and Knowledge Gap Detection

### Agent Responsibilities (all in `src/agents/`)

| Agent | File | Purpose |
|-------|------|---------|
| **Query Intent Agent** | `query_intent_agent.py` | Classifies incoming queries (intent: simple search, complex reasoning, gap analysis, etc.) and routes to appropriate pipeline. |
| **Retrieval Agent** | `retrieval_agent.py` | Performs semantic search against Chroma vector store; returns top-k documents. |
| **Knowledge Gap Agent** | `knowledge_gap_agent.py` | Analyzes retrieved documents against query requirements; detects missing fields or information gaps. |
| **Vector Agent** | `vector_agent.py` | Helper agent wrapping vector store operations (ingest, query) for multi-agent use. |
| **Context Builder Agent** | `context_builder_agent.py` | Converts retrieved documents and metadata into formatted, LLM-readable prompt blocks. |
| **CLARA Agent** | `clara_agent.py` | Multi-step decomposition and iterative reasoning for complex queries; orchestrates sub-queries and chain-of-thought. |
| **Final Answer Agent** | `final_answer_agent.py` | Formats LLM outputs into final response for end user; handles summarization and quality checks. |

### Orchestrator (what it does)

File: `src/orchestrator/orchestrator.py`

- **Receives** a user query from CLI or API.
- **Routes** through agents:
  1. Calls `query_intent_agent` to determine query type (simple, complex, gap-detection, etc.).
  2. For retrieval: invokes `retrieval_agent` to fetch relevant documents.
  3. For gap detection: calls `knowledge_gap_agent` to identify missing data.
  4. For complex reasoning: delegates to `clara_system` for multi-step decomposition.
  5. Builds LLM prompt via `context_builder_agent`.
  6. Calls LLM (`src/llm_engine/ollama_client.py` or other provider).
  7. Formats final output via `final_answer_agent`.

### Knowledge Gap Detection (workflow)

1. **Retrieval Phase**: Retrieve top-k documents from Chroma for a given query.
2. **Inspection Phase**: `knowledge_gap_agent` compares retrieved fields/columns against:
   - Query intent (what info was asked for?).
   - Required schema (what fields should be present?).
   - Retrieved metadata keys.
3. **Gap Report**: If fields are missing:
   - Return a partial answer with confidence score.
   - Suggest user provide more data or rephrase query.
   - Optionally trigger `clara_system` to attempt inference with uncertainty warnings.

**Example gap scenario**:
- Query: "Which flights depart before 6 AM?"
- Retrieved docs: have `departure_time`, but no `airline` or `price` information.
- Gap detected: fields `airline`, `price` missing; can still answer departure time query but note data incompleteness.

### CLARA System (advanced reasoning)

Files: `src/reasoning/clara_agent.py`, `src/agents/clara_agent.py`

CLARA (Compositional Layer for Agentic Reasoning Architecture) is used for **complex, multi-step queries**:

1. **Decomposition**: Break a complex query into sub-steps (e.g., "Find cheapest morning flights from LAX" → step1: filter by origin, step2: filter by time, step3: sort by price).
2. **Iterative retrieval**: Run multiple retrieval calls with refined queries.
3. **Aggregation**: Combine results from each step.
4. **LLM-driven planning**: Use the LLM to decide next steps based on intermediate results.

**Invocation**: Orchestrator calls CLARA when `query_intent_agent` flags the query as complex or when gap detection suggests iterative retrieval is needed.

---

## System Architecture Diagram

```text
┌─────────────┐
│ User Query  │ (CLI / API)
└──────┬──────┘
       │
       v
┌─────────────────────────────┐
│  Orchestrator               │
│  (src/orchestrator/)        │
└──────────┬──────────────────┘
           │
    ┌──────┴──────────────────┐
    │                         │
    v                         v
┌─────────────┐      ┌──────────────────┐
│ Intent      │      │ Routing decision │
│ Agent       │──────► (simple/complex/ │
└─────────────┘      │  gap)            │
                     └────┬─────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
    SIMPLE            COMPLEX           GAP
        │                 │              ANALYSIS
        │                 │                 │
        v                 v                 v
   ┌────────────┐   ┌──────────┐    ┌────────────────┐
   │ Retrieval  │   │ CLARA    │    │ Knowledge Gap  │
   │ Agent      │   │ System   │    │ Agent          │
   └──────┬─────┘   │ (multi-  │    └───────┬────────┘
          │         │  step)   │            │
          │         └────┬─────┘            │
          │              │                  │
          │         ┌────────────┐          │
          │         │ Retrieval  │          │
          │         │ (iterated) │          │
          │         └────┬───────┘          │
          │              │                  │
          └──────┬───────┴──────────────────┘
                 │
                 v
        ┌────────────────────┐
        │ Vector Store       │
        │ (Chroma)           │
        └────────┬───────────┘
                 │
          ┌──────┴──────┐
          │ Documents,  │
          │ Metadata,   │
          │ Embeddings  │
          └─────────────┘
          
        Retrieved Docs
                 │
                 v
        ┌────────────────────┐
        │ Context Builder    │
        │ Agent              │
        └────────┬───────────┘
                 │
                 v
        ┌────────────────────┐
        │ LLM Engine         │
        │ (Ollama client or  │
        │  other provider)   │
        └────────┬───────────┘
                 │
                 v
        ┌────────────────────┐
        │ Final Answer       │
        │ Agent              │
        └────────┬───────────┘
                 │
                 v
        ┌────────────────────┐
        │ Final Response     │
        │ to User            │
        └────────────────────┘
```

---

## Changing LLM or CSV Locations (quick pointers)

### Switch LLM provider or model

- **Config file**: `src/config.py` — change `LLM_Provider`, `OLLAMA_MODEL`, `OLLAMA_BASE_URL`.
- **Factory selection**: `src/llm_engine/factory.py` — add new provider or switch logic.
- **Ollama-specific**: `src/llm_engine/ollama_client.py` (line 9) — instantiation and method calls.
- **Environment override**: Set `.env` variables for runtime configuration.

### Change CSV ingestion path

- **Default path**: `reingest.py` (line 6) — `DEFAULT_CSV = os.path.join("data", "flights.csv")`.
- **Ingestion logic**: `src/vector_store.py` (method `ingest_csv`) — handles `pd.read_csv`, row → text conversion, embedding, and Chroma storage.
- **Programmatic use**: Import `FlightVectorStore` and call `ingest_csv(path)` directly.

---

## Troubleshooting (concise)

| Issue | Fix |
|-------|-----|
| Ollama connection error | Ensure `ollama serve` is running; check `OLLAMA_BASE_URL` is correct. |
| Model not found | Run `ollama pull <model-name>`. |
| Chroma DB errors | Delete `vector_db/` directory and re-run `python reingest.py`. |
| Slow embedding generation | Use a smaller model (e.g., `nomic-embed-text` instead of large models). |
| Python import errors | Ensure you're in the virtual environment (`source kbBot/bin/activate`) and dependencies are installed (`pip install -r requirements.txt`). |

---

## Development and Testing

### Run project structure inspection

```bash
python project_structure.py
```

### (Optional) Run tests

```bash
pytest tests/
```

---

## Additional Resources

- **Ollama docs**: https://ollama.com/docs
- **Chroma docs**: https://docs.trychroma.com
- **SentenceTransformers docs**: https://www.sbert.net
- **LangChain community**: https://github.com/langchain-ai/langchain-community

---

_README updated to include generic CSV ingestion, step-by-step setup, agent responsibilities, orchestrator flow, gap detection explanation, CLARA system overview, Ollama model pulling guide, and system architecture diagram._