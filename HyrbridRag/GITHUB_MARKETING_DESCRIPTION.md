# HybridRAG for Semiconductor Specification Intelligence

## GitHub About Description

Offline hybrid RAG system for querying SPI semiconductor specifications with vector search, BM25, and a structured knowledge graph.

## Repository Tagline

Turn dense hardware protocol PDFs into a searchable, citeable, graph-grounded engineering assistant.

## Detailed Project Description

HybridRAG is a precision-first retrieval system built for semiconductor and embedded-systems documentation. It converts SPI protocol specification PDFs into a hybrid knowledge layer that combines semantic vector search, lexical BM25 retrieval, and a structured graph extracted from the document ontology.

Unlike a basic PDF chatbot, this project is designed for engineering-grade questions where exact details matter: register names, bit fields, reset values, SPI modes, timing behavior, page-level provenance, and cross-reference relationships. The system is especially useful for ASIC verification, embedded firmware development, protocol bring-up, and technical documentation analysis.

The project runs locally with Ollama, Chroma, Docling, and docling-graph, making it suitable for private datasheets or internal hardware specifications that cannot be sent to cloud APIs.

## What It Does

- Parses SPI specification PDFs into layout-aware document chunks.
- Builds a hybrid vector index using Chroma embeddings plus BM25 keyword retrieval.
- Preserves atomic register and bit-field table chunks to avoid losing hardware-level precision.
- Extracts structured entities such as registers, bit fields, timing parameters, state machines, SPI modes, and frame formats.
- Builds a knowledge graph for exact lookup and relationship traversal.
- Provides a tool-calling agent that can choose vector search, graph search, or both.
- Enforces citation-aware answers using chunk metadata and page provenance.
- Includes an audit script for ingestion quality, graph topology, and vector-to-graph grounding checks.
- Runs offline using local Ollama models.

## Why This Matters

Hardware specifications are difficult for standard RAG systems because they mix prose, tables, register maps, timing rules, and protocol state behavior. Pure vector search can miss exact symbols like `CPOL`, `CPHA`, `SPIxCON0`, or bit ranges. Pure graph extraction can miss explanatory context.

This project combines both approaches:

- Vector retrieval answers conceptual and explanatory questions.
- BM25 protects exact-match technical terms.
- The knowledge graph supports structured lookup and relationship traversal.
- Provenance metadata keeps answers grounded in the original specification.

The result is a stronger foundation for verification workflows, including future test-plan generation, coverage planning, assertion drafting, and register-level scenario discovery.

## Ideal Use Cases

- ASIC and SoC verification teams querying protocol specifications.
- Embedded firmware engineers searching register and bit-field behavior.
- Hardware documentation teams validating spec consistency.
- RAG developers building domain-specific retrieval systems for technical PDFs.
- Engineers prototyping offline AI assistants for private datasheets.

## Key Features

### Hybrid Retrieval

Combines dense semantic search with BM25 lexical search using reciprocal-rank fusion. This improves both natural-language understanding and exact technical symbol retrieval.

### Graph-Grounded Extraction

Uses a Pydantic ontology and docling-graph to extract structured SPI entities into a NetworkX graph.

### Atomic Table Chunking

Keeps register and bit-field tables together so retrieval does not split critical hardware facts across unrelated chunks.

### Offline Local Execution

Uses Ollama for local chat, embeddings, and graph extraction. This is useful for confidential or internal specifications.

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

`rag`, `hybrid-rag`, `semiconductor`, `asic-verification`, `embedded-systems`, `spi`, `knowledge-graph`, `vector-search`, `bm25`, `ollama`, `chromadb`, `docling`, `pydantic`, `networkx`, `offline-ai`

## Short README Intro

HybridRAG is an offline AI assistant for semiconductor protocol specifications. It combines vector search, BM25, and a structured knowledge graph to answer detailed SPI questions with page-grounded citations. The system is built for engineering workflows where exact register names, bit fields, timing rules, and protocol behavior must be retrieved accurately from dense technical PDFs.

## Longer README Intro

HybridRAG is a local-first retrieval system for semiconductor and embedded protocol specifications. It transforms SPI datasheets and application notes into a hybrid search layer made of semantic chunks, lexical keyword retrieval, and a structured knowledge graph.

The goal is to make hardware specifications easier to query without losing the precision required for ASIC verification and firmware development. A user can ask conceptual questions such as how SPI transfer sequencing works, or exact questions such as which register field controls a mode bit. The agent can use vector search for explanatory context, graph search for structured facts, and provenance metadata for citations back to the source document.

Because the pipeline runs locally with Ollama, it can be used with private or internal specifications without sending document content to an external API.

## Elevator Pitch

HybridRAG turns hardware protocol PDFs into a local, citeable engineering assistant. It combines semantic search, keyword retrieval, and graph extraction so teams can ask precise questions about registers, bit fields, timing behavior, and SPI protocol details without losing source-grounded traceability.

## Demo Script

1. Add an SPI specification PDF to `data/raw_specs/`.
2. Run `python ingest.py` to build the vector index and knowledge graph.
3. Run `python audit_hybrid_quality.py` to check chunk quality and graph retrievability.
4. Run `python main.py` and ask questions such as:
   - What does `SPIxCON0` control?
   - Which fields affect SPI transfer mode?
   - Explain SPI master/slave communication flow.
   - Where is `BMODE` described?

## Positioning Statement

HybridRAG is not just another document chatbot. It is a domain-aware retrieval architecture for hardware specifications, built around the reality that engineering documents contain both natural language and exact structured facts. By combining vector retrieval, BM25, and graph topology, it provides a stronger foundation for trusted technical Q&A and future verification automation.
