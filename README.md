# HybridRAG for Semiconductor Datasheets

HybridRAG is an offline, local-first retrieval augmented generation (RAG)
system for semiconductor chip datasheets and embedded hardware specifications.
It combines semantic vector search, BM25 keyword retrieval, vision-derived
technical image summaries, and a structured knowledge graph so engineers can
query dense datasheet PDFs with grounded citations.

The current configuration targets the Infineon AURIX TC38x /
SAK-TC389QP-160F400S AE datasheet, but the ingestion pipeline is designed to
support additional microcontroller, ASIC, SoC, peripheral, interface, and
board-level specification PDFs.

## Repository Description

This project turns semiconductor datasheets into a searchable engineering
assistant. It is built for questions where exact details matter: part numbers,
package variants, pin functions, peripheral signals, register names, register
addresses, bit fields, reset values, timing behavior, electrical limits,
interface modes, frame formats, diagrams, and page-level provenance.

Unlike a basic PDF chatbot, HybridRAG uses multiple retrieval layers:

- ChromaDB vector search for semantic questions and explanatory context.
- BM25 lexical search for exact technical symbols, register names, pin names,
  package codes, and electrical parameter names.
- Atomic table chunks for register maps and bit-field tables.
- Local vision-model summaries for timing diagrams, block diagrams,
  flowcharts, package drawings, charts, and schematics.
- A docling-graph knowledge graph for structured entity lookup and
  relationship traversal.
- Citation-aware agent responses grounded in retrieved document chunks.

Because it runs locally with Ollama, Docling, ChromaDB, NetworkX, Pydantic,
and docling-graph, it can be used with private or internal hardware
specifications without sending document content to external APIs.

## Keywords

Hybrid RAG, semiconductor datasheet RAG, offline RAG, local RAG, retrieval
augmented generation, vector search, BM25, ChromaDB, Ollama, Docling,
docling-graph, knowledge graph, technical PDF retrieval, ASIC verification,
SoC verification, embedded systems, register map search, bit-field search,
pinout search, hardware specification intelligence, engineering document AI.

## Why Hybrid Retrieval

Semiconductor datasheets mix prose, tables, diagrams, symbols, pin mappings,
package variants, timing parameters, and electrical characteristics. Pure
vector search can miss exact strings such as `ENDINIT`, `P20.3`,
`SAK-TC389QP`, or a register address. Pure graph extraction can miss the
surrounding explanation that gives a fact engineering meaning.

HybridRAG keeps both retrieval paths available:

- Vector retrieval answers conceptual questions such as how a peripheral works.
- BM25 retrieval protects exact terms, acronyms, symbols, and register fields.
- Graph retrieval supports exact lookup of structured datasheet entities.
- Image summaries make technical diagrams searchable alongside text.
- Provenance metadata ties answers back to the original document page range.

The agent can call `search_vector`, `search_graph`, or both for the same user
question.

## Features

- Offline datasheet Q&A using local Ollama models.
- Layout-aware PDF parsing with Docling.
- Hybrid ChromaDB vector search and BM25 retrieval with reciprocal-rank fusion.
- Register and bit-field table preservation through atomic chunks.
- Structured graph extraction using a Pydantic datasheet ontology.
- NetworkX graph fusion across multiple specification documents.
- Cached local analysis of technical images and diagrams.
- CLI chatbot for source-grounded questions.
- Ingestion audit script for chunk quality, graph retrievability, and
  vector-to-graph grounding.
- Configurable document registry, model names, chunking, retrieval, and agent
  behavior in one place.

## Example Questions

After indexing a datasheet, you can ask questions such as:

- What package variants are listed for SAK-TC389QP-160F400S AE?
- Which pins expose a selected peripheral signal?
- What does the ENDINIT-related register field control?
- What are the operating voltage or temperature limits?
- Which section describes a timing diagram for a selected interface?
- What reset value is associated with a register or bit field?
- Which peripherals are connected to a given interface mode?

## Architecture

```text
PDF datasheets
    |
    v
Docling parser
    |
    +--> structural text chunks
    +--> atomic register and bit-field table chunks
    +--> extracted technical images
    |
    v
Local Ollama models
    |
    +--> embeddings for ChromaDB
    +--> chat and extraction model calls
    +--> vision summaries for technical diagrams
    |
    v
Hybrid index
    |
    +--> Chroma vector store
    +--> BM25 lexical index
    +--> docling-graph / NetworkX knowledge graph
    |
    v
Tool-calling agent with citations
```

## Project Layout

```text
agent.py                      Tool-calling answer loop with citation controls
audit_hybrid_quality.py       Offline quality checks for chunks and graph data
config.py                     Paths, document registry, models, and tunables
docling_reader.py             PDF parsing, chunking, table handling, image chunks
document_index.py             ChromaDB and BM25 hybrid vector index
graph_builder.py              docling-graph extraction and graph fusion
image_processor.py            Cached local vision analysis for technical images
ingest.py                     One-time indexing pipeline
llm_backend.py                Ollama chat and embedding wrapper
main.py                       CLI chatbot entry point
progress.py                   Progress logging helpers
schemas/datasheet_ontology.py Pydantic ontology for datasheet entities
tools.py                      Agent tools: vector search, graph search, helpers
requirements.txt              Python dependency list
```

Generated data is intentionally ignored by git:

```text
data/raw_specs/      Local datasheet PDFs
data/chroma_store/   ChromaDB persistence
data/graph_store/    Extracted/fused graph artifacts
data/cache/          Parsed-document and image-analysis cache
```

## Requirements

- Python 3.11 or newer is recommended.
- Ollama installed and running locally.
- Enough disk space for datasheet PDFs, ChromaDB indexes, graph artifacts, and
  image caches.
- A local CPU/GPU setup capable of running the configured Ollama models.

Install or pull the default local models:

```bash
ollama pull gpt-oss:20b
ollama pull phi4
ollama pull nomic-embed-text
ollama pull qwen2.5-coder:7b
ollama pull qwen2.5vl
```

If your machine has limited VRAM, replace the model names in `config.py` with
smaller Ollama models that fit your hardware.

## Setup

Clone the repository:

```bash
git clone https://github.com/Pranavindia2007/HybridRag.git
cd HybridRag
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Add datasheet PDFs to `data/raw_specs/`. Then update `DOCUMENT_REGISTRY` in
`config.py` so each `doc_id` points to the correct PDF file:

```python
DOCUMENT_REGISTRY = {
    "sak_tc389qp_160f400s_ae": RAW_DOCS_DIR / "infineon-tc38x-datasheet-en.pdf",
}
```

## Ingest Datasheets

Build the vector index, image summaries, graph extraction, and fused graph:

```bash
python ingest.py
```

Useful ingest options:

```bash
python ingest.py --skip-images
python ingest.py --image-cache-only
```

The ingest step can take time because it parses large PDFs, embeds chunks,
analyzes technical images, and extracts structured graph data with local LLM
models.

## Run the Chatbot

After ingestion completes:

```bash
python main.py
```

Inside the CLI:

```text
You: Which pins expose the selected SPI signal?
You: reset
You: exit
```

## Audit Retrieval Quality

Run the offline quality checks:

```bash
python audit_hybrid_quality.py
```

Check specific graph/vector grounding terms:

```bash
python audit_hybrid_quality.py --terms SAK-TC389QP ENDINIT SCU CAN ADC
```

Export a JSON report:

```bash
python audit_hybrid_quality.py --skip-reparse --json data/audit_report.json
```

The audit checks ingestion quality, metadata completeness, graph topology,
entity retrievability, and alignment between graph terms and vector chunks.

## Development Workflow

This repository is structured for normal git-based development:

```bash
git status
git checkout -b feature/your-change
python ingest.py --skip-images
python audit_hybrid_quality.py --skip-reparse
git add .
git commit -m "Describe your change"
git push origin feature/your-change
```

Recommended development practices:

- Keep source code, schemas, docs, and small examples in git.
- Keep datasheet PDFs, generated vector stores, graph files, caches,
  virtual environments, and Python bytecode out of git.
- Use feature branches for changes and pull requests for review.
- Run the audit script after retrieval, chunking, graph, or configuration
  changes.
- Update this README whenever setup, model choices, or expected workflows
  change.

## Configuration

Most project settings live in `config.py`:

- `DOCUMENT_REGISTRY` controls which datasheets are indexed.
- `CHAT_MODEL`, `CHAT_MODEL_FALLBACK`, `EMBED_MODEL`,
  `GRAPH_EXTRACTION_MODEL`, and `VISION_MODEL` control Ollama model names.
- `CHUNK_CONFIG` controls structural chunk size, overlap, and table handling.
- `IMAGE_PROCESSING_CONFIG` controls technical image extraction and caching.
- `RETRIEVAL_CONFIG` controls dense/BM25 top-k values and fusion weights.
- `AGENT_CONFIG` controls tool-call limits, temperature, and citation behavior.

## Current Limitations

- Register and bit-field table detection is heuristic and should be validated
  against each new datasheet family.
- There is no cross-encoder reranker yet; retrieval currently uses
  reciprocal-rank fusion over dense and BM25 results.
- There is no full automated golden-answer evaluation suite yet.
- Large datasheets can make ingestion slow on smaller machines.
- This repository does not include proprietary or copyrighted datasheet PDFs.

## Roadmap Ideas

- Add a small golden-question retrieval evaluation harness.
- Add optional reranking for fused vector/BM25 results.
- Add a web UI or API server on top of the existing agent.
- Add export tools for graph entities, register maps, and pin tables.
- Add scenario-generation workflows for ASIC and SoC verification.
- Add richer provenance views for page snippets, diagrams, and table cells.

## Suggested GitHub Topics

`rag`, `hybrid-rag`, `offline-rag`, `semiconductor`, `datasheet`,
`asic-verification`, `soc-verification`, `embedded-systems`,
`knowledge-graph`, `vector-search`, `bm25`, `ollama`, `chromadb`,
`docling`, `docling-graph`, `pydantic`, `networkx`, `technical-pdf`.

## License

No license file is currently included. Add a license before publishing this
project for broader reuse.
