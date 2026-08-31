"""
ingest.py

Run this once (and again whenever specs change) to build both halves of
the hybrid index:
  1. Chunk each spec (docling_reader) -> embed + index (document_index)
  2. Extract each spec into the graph ontology (graph_builder) -> fuse

Usage:
    python ingest.py
"""

from __future__ import annotations
import pickle
from pathlib import Path

from config import DOCUMENT_REGISTRY, GRAPH_STORE_DIR
from docling_reader import build_chunks
from document_index import HybridIndex
from graph_builder import build_full_graph
from llm_backend import OllamaBackend


FUSED_GRAPH_PATH = GRAPH_STORE_DIR / "fused_graph.gpickle"


def main():
    print("\n== Ingest Kick In ==")
    backend = OllamaBackend()
    backend.ensure_models_available()
    print("\n== Ollma Models Available ==")
    index = HybridIndex(backend)

    print("== Vector indexing ==")
    for doc_id, pdf_path in DOCUMENT_REGISTRY.items():
        if not Path(pdf_path).exists():
            print(f"  [skip] {doc_id}: {pdf_path} not found")
            continue
        print(f"  Parsing + chunking {doc_id} ...")
        chunks = build_chunks(doc_id, str(pdf_path))
        atomic = sum(1 for c in chunks if c.chunk_type == "atomic_table")
        structural = sum(1 for c in chunks if c.chunk_type == "structural")
        print(f"    {len(chunks)} chunks ({atomic} atomic table, {structural} structural)")
        index.add_chunks(chunks)
        print(f"    Indexed {doc_id}.")

    print("\n== Graph extraction + fusion ==")
    graph = build_full_graph()
    print(f"  Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    with open(FUSED_GRAPH_PATH, "wb") as f:
        pickle.dump(graph, f)
    print(f"  Saved to {FUSED_GRAPH_PATH}")

    print("\nDone. Run `python main.py` to chat.")


if __name__ == "__main__":
    main()
