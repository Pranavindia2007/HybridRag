"""
ingest.py

Run this once (and again whenever datasheets change) to build both halves of
the hybrid index:
  1. Chunk each datasheet (docling_reader) -> embed + index (document_index)
  2. Extract each datasheet into the graph ontology (graph_builder) -> fuse

Usage:
    python ingest.py
"""

from __future__ import annotations
import argparse
import pickle
from pathlib import Path

from config import (
    DOCUMENT_REGISTRY,
    EMBED_MODEL,
    GRAPH_EXTRACTION_MODEL,
    GRAPH_STORE_DIR,
    IMAGE_PROCESSING_CONFIG,
)
from docling_reader import build_chunks
from document_index import HybridIndex
from graph_builder import build_full_graph
from llm_backend import OllamaBackend
from progress import progress_log


FUSED_GRAPH_PATH = GRAPH_STORE_DIR / "datasheet_fused_graph.gpickle"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build datasheet vector and graph indexes.")
    parser.add_argument(
        "--image-cache-only",
        action="store_true",
        help="Reuse cached image analyses and skip new Ollama vision calls.",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Skip image extraction/analysis for this ingest run.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    show_progress = True
    print("\n== Ingest Kick In ==")
    progress_log("Checking local Ollama models ...", show_progress)
    backend = OllamaBackend()
    required_models = [backend.chat_model, EMBED_MODEL, GRAPH_EXTRACTION_MODEL]
    process_images = IMAGE_PROCESSING_CONFIG.enabled and not args.skip_images
    analyze_uncached_images = process_images and not args.image_cache_only
    if args.skip_images:
        progress_log("Image processing disabled for this run.", show_progress)
    elif args.image_cache_only:
        progress_log("Image processing is cache-only; new vision calls will be skipped.", show_progress)
    if analyze_uncached_images:
        required_models.append(IMAGE_PROCESSING_CONFIG.vision_model)
    backend.ensure_models_available(required_models)
    print("\n== Ollama Models Available ==")
    index = HybridIndex(backend)

    print("== Vector indexing ==")
    for doc_id, pdf_path in DOCUMENT_REGISTRY.items():
        if not Path(pdf_path).exists():
            print(f"  [skip] {doc_id}: {pdf_path} not found")
            continue
        print(f"  Parsing + chunking {doc_id} ...")
        chunks = build_chunks(
            doc_id,
            str(pdf_path),
            backend=backend,
            process_images=process_images,
            analyze_uncached_images=analyze_uncached_images,
            show_progress=show_progress,
        )
        atomic = sum(1 for c in chunks if c.chunk_type == "atomic_table")
        structural = sum(1 for c in chunks if c.chunk_type == "structural")
        images = sum(1 for c in chunks if c.chunk_type == "image_analysis")
        print(
            f"    {len(chunks)} chunks "
            f"({atomic} atomic table, {structural} structural, {images} image analysis)"
        )
        index.add_chunks(chunks, show_progress=show_progress)
        print(f"    Indexed {doc_id}.")

    print("\n== Graph extraction + fusion ==")
    graph = build_full_graph(show_progress=show_progress)
    print(f"  Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    with open(FUSED_GRAPH_PATH, "wb") as f:
        pickle.dump(graph, f)
    print(f"  Saved to {FUSED_GRAPH_PATH}")

    print("\nDone. Run `python main.py` to chat.")


if __name__ == "__main__":
    main()
