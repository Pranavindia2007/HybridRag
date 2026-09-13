# HybridRAG for Semiconductor Specification Intelligence

## GitHub About Description

Offline hybrid RAG for semiconductor datasheets with Docling, ChromaDB, BM25, Ollama vision summaries, and a knowledge graph.

## Repository Tagline

Turn dense chip datasheets into a searchable, citeable, graph-grounded engineering assistant.

## Detailed Project Description

HybridRAG is a precision-first retrieval system built for semiconductor and embedded-systems documentation. It converts chip datasheet PDFs into a hybrid knowledge layer that combines semantic vector search, lexical BM25 retrieval, technical image analysis, and a structured graph extracted from the document ontology.

Unlike a basic PDF chatbot, this project is designed for engineering-grade questions where exact details matter: part numbers, packages, pin functions, register names, bit fields, reset values, timing behavior, electrical limits, page-level provenance, and cross-reference relationships. The system is useful for ASIC/SoC verification, embedded firmware development, board bring-up, and technical documentation analysis.

The project runs locally with Ollama, Chroma, Docling, and docling-graph, making it suitable for private datasheets or internal hardware specifications that cannot be sent to cloud APIs.

## What It Does

- Parses semiconductor datasheet PDFs into layout-aware document chunks.
- Builds a hybrid vector index using Chroma embeddings plus BM25 keyword retrieval.
- Preserves atomic register and bit-field table chunks to avoid losing hardware-level precision.
- Extracts structured entities such as packages, pins, peripherals, registers, bit fields, timing parameters, electrical characteristics, state machines, interface modes, and frame formats.
- Processes technical visuals such as flowcharts, timing diagrams, block diagrams, package/pin diagrams, charts, and schematics through a cached local vision model.
- Builds a knowledge graph for exact lookup and relationship traversal.
- Provides a tool-calling agent that can choose vector search, graph search, or both.
- Enforces citation-aware answers using chunk metadata and page provenance.
- Includes an audit script for ingestion quality, graph topology, and vector-to-graph grounding checks.
- Runs offline using local Ollama models.

## Why This Matters

Hardware datasheets are difficult for standard RAG systems because they mix prose, tables, register maps, package tables, pinouts, timing rules, electrical characteristics, and diagrams. Pure vector search can miss exact symbols like part codes, pin names, register fields, addresses, or bit ranges. Pure graph extraction can miss explanatory context.

This project combines both approaches:

- Vector retrieval answers conceptual and explanatory questions.
- BM25 protects exact-match technical terms.
- The knowledge graph supports structured lookup and relationship traversal.
- Provenance metadata keeps answers grounded in the original datasheet.

The result is a stronger foundation for verification workflows, including future test-plan generation, coverage planning, assertion drafting, and register-level scenario discovery.

## Ideal Use Cases

- ASIC and SoC verification teams querying chip datasheets.
- Embedded firmware engineers searching register and bit-field behavior.
- Hardware documentation teams validating spec consistency.
- RAG developers building domain-specific retrieval systems for technical PDFs.
- Engineers prototyping offline AI assistants for private datasheets.

## Key Features

### Hybrid Retrieval

Combines dense semantic search with BM25 lexical search using reciprocal-rank fusion. This improves both natural-language understanding and exact technical symbol retrieval.

### Graph-Grounded Extraction

Uses a Pydantic ontology and docling-graph to extract structured datasheet entities into a NetworkX graph.

### Atomic Table Chunking

Keeps register and bit-field tables together so retrieval does not split critical hardware facts across unrelated chunks.

### Offline Local Execution

Uses Ollama for local chat, embeddings, graph extraction, and vision-based image analysis. This is useful for confidential or internal datasheets.

### Retrieval Quality Auditing

Includes `audit_hybrid_quality.py` to test:

- ingestion and chunk quality
- graph topology retrievability
- vector-to-graph alignment
- grounding of graph terms in vector chunks

### Citation-First Agent Loop

The agent is designed to answer only from retrieved evidence, reducing unsupported guesses.

## Tech Stack

- Python
- Ollama
- ChromaDB
- BM25 / rank-bm25
- Docling
- docling-graph
- Pydantic
- NetworkX

## Suggested GitHub Topics

`rag`, `hybrid-rag`, `offline-rag`, `semiconductor`, `datasheet`, `datasheet-rag`, `graph-rag`, `knowledge-graph`, `vector-search`, `bm25`, `chromadb`, `ollama`, `docling`, `docling-graph`, `technical-pdf`, `asic-verification`, `soc-verification`, `embedded-systems`, `retrieval-augmented-generation`, `ai-engineering`

## Short README Intro

HybridRAG is an offline AI assistant for semiconductor chip datasheets. It combines vector search, BM25, vision-derived image summaries, and a structured knowledge graph to answer detailed datasheet questions with page-grounded citations. The system is built for engineering workflows where exact part numbers, package variants, pin functions, register names, bit fields, timing rules, and electrical limits must be retrieved accurately from dense technical PDFs.

## Longer README Intro

HybridRAG is a local-first retrieval system for semiconductor and embedded datasheets. It transforms chip datasheets and application notes into a hybrid search layer made of semantic chunks, lexical keyword retrieval, technical image summaries, and a structured knowledge graph.

The goal is to make hardware datasheets easier to query without losing the precision required for ASIC verification and firmware development. A user can ask conceptual questions about a peripheral, or exact questions such as which package exposes a signal, what a register field controls, or what an electrical limit is under specific conditions. The agent can use vector search for explanatory context, graph search for structured facts, image summaries for diagrams, and provenance metadata for citations back to the source document.

Because the pipeline runs locally with Ollama, it can be used with private or internal specifications without sending document content to an external API.

## Elevator Pitch

HybridRAG turns chip datasheet PDFs into a local, citeable engineering assistant. It combines semantic search, keyword retrieval, image analysis, and graph extraction so teams can ask precise questions about part variants, pins, packages, peripherals, registers, bit fields, timing behavior, and electrical characteristics without losing source-grounded traceability.

## Demo Script

1. Add a semiconductor datasheet PDF to `data/raw_specs/`.
2. Run `python ingest.py` to build the vector index and knowledge graph.
3. Run `python audit_hybrid_quality.py` to check chunk quality and graph retrievability.
4. Run `python main.py` and ask questions such as:
   - What package variants are listed for SAK-TC389QP-160F400S AE?
   - Which pins expose a selected peripheral signal?
   - What does the ENDINIT-related register field control?
   - What are the operating voltage or temperature limits?

## Positioning Statement

HybridRAG is not just another document chatbot. It is a domain-aware retrieval architecture for hardware specifications, built around the reality that engineering documents contain both natural language and exact structured facts. By combining vector retrieval, BM25, and graph topology, it provides a stronger foundation for trusted technical Q&A and future verification automation.
