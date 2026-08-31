# SPI Spec RAG (hybrid vector + graph, offline)

Precision-first RAG chatbot over one or two SPI ASIC verification specs,
built to run entirely offline on a consumer GPU. Foundation for a later
test-scenario-generation tool -- don't extend this into that tool by
bolting more code onto `main.py`; build a new module on top of
`SPIRagAgent` + `HybridIndex` instead.

## Why hybrid (vector + graph)

[docling-graph](https://github.com/docling-project/docling-graph) turns a
parsed document into a validated knowledge graph with automatic
provenance -- but it does **not** do embeddings or vector retrieval. So:

- **Vector store (Chroma + BM25)** — dense + lexical hybrid search over
  chunked spec text. This is what answers "how does X work" / "explain Y".
- **Knowledge graph (docling-graph)** — structured registers, bit fields,
  timing parameters, SPI modes, extracted into a Pydantic-defined ontology
  and fused across both specs. This is what answers "what's the reset
  value of bit 3 in SPI_CR1" precisely, and lets you compare the same
  register across two specs via graph traversal instead of guesswork.

The agent picks between `search_vector` and `search_graph` (and can use
both) per question.

## Setup

1. **Install Ollama** and pull the local models:
   ```bash
   ollama pull qwen2.5:7b-instruct    # chat / extraction model
   ollama pull phi4                    # fallback if VRAM constrained
   ollama pull nomic-embed-text        # embeddings
   ```
   Check your GPU's VRAM against these before committing -- drop to a
   smaller quant (e.g. `qwen2.5:7b-instruct-q4_0`) if `qwen2.5:7b-instruct`
   OOMs, and update `CHAT_MODEL` / `GRAPH_EXTRACTION_MODEL` in `config.py`.

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Drop your spec PDFs** into `data/raw_specs/`, named to match
   `DOCUMENT_REGISTRY` in `config.py` (or edit the registry to match your
   filenames). Works with one spec too -- just leave the second entry's
   file missing and ingestion will skip it.

4. **Build the indices** (parses both specs, chunks, embeds, extracts the
   graph, fuses):
   ```bash
   python ingest.py
   ```
   This step calls the local LLM for graph extraction, so expect it to
   take a few minutes per spec depending on length and your GPU.

5. **Chat**:
   ```bash
   python main.py
   ```

## Tuning bit-level accuracy first

Before trusting this on real questions, check `ingest.py`'s printed chunk
counts (`atomic table` vs `structural`). If your spec's bit-field tables
aren't being detected:

- Open `docling_reader.py::_looks_like_bitfield_table` and check the
  header-keyword heuristic against your spec's actual table headers.
- Open `docling_reader.py::_table_to_atomic_chunks` and check the
  register-grouping logic matches how your spec lays out register/field
  rows (some specs put the register name in a merged header row above
  the field table rather than a column -- that needs a small tweak here).

Getting this right matters more than any retrieval-layer tuning: if a
bit-field table gets misparsed into structural (prose) chunks, no amount
of embedding-model improvement will recover it.

## Project layout

```
config.py            All paths, models, chunking/retrieval tunables
docling_reader.py     Docling parsing -> structural + atomic table chunks
document_index.py     Chroma + BM25 hybrid vector index (RRF fusion)
schemas/
  spi_ontology.py      Pydantic templates for docling-graph extraction
graph_builder.py      docling-graph extraction + cross-spec fusion
llm_backend.py         Ollama chat + embedding wrapper
tools.py               Agent tool definitions (search_vector, search_graph, ...)
agent.py               Tool-calling loop, hard-stopped, citation-enforced
ingest.py              One-time indexing pipeline
main.py                CLI chatbot
```

## Known gaps to close before the test-scenario-generation phase

- `_looks_like_bitfield_table` / `_table_to_atomic_chunks` are heuristic
  and need a validation pass against your actual spec's table structure
  (see above) -- run `ingest.py`, spot-check a handful of atomic chunks
  by hand before trusting retrieval.
- No reranker model yet (RRF fusion only). If dense+BM25 fusion isn't
  precise enough once you test against real questions, add a cross-encoder
  rerank step in `document_index.py::search` over the fused top-N.
- No automated retrieval eval harness yet -- worth building a small golden
  set (question -> expected chunk_id/register) before scaling up, same
  approach as the CI-style eval harness discussed for the generic
  multi-protocol pipeline.
