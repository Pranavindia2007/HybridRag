"""
graph_builder.py

Wraps docling-graph to build the structured/relational half of the hybrid
index. Runs the DatasheetDocument Pydantic template
(schemas/datasheet_ontology.py) against each datasheet, then uses
docling-graph's built-in fusion to merge documents into one cross-document
graph.

Every node carries docling-graph's automatic provenance (source chunk +
page), so graph answers stay just as citable as vector answers.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional

import networkx as nx
from docling_graph import run_pipeline, PipelineConfig
from docling_graph.core.converters.graph_converter import GraphConverter
from docling_graph.llm_clients.config import LlmRuntimeOverrides

from config import GRAPH_STORE_DIR, GRAPH_EXTRACTION_MODEL, DOCUMENT_REGISTRY
from schemas.datasheet_ontology import DatasheetDocument
from progress import ProgressBar, progress_log


def extract_graph_for_doc(doc_id: str, pdf_path: str, show_progress: bool = False) -> nx.Graph:
    """Run the docling-graph pipeline for a single datasheet document."""
    out_dir = GRAPH_STORE_DIR / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline_config = PipelineConfig(
        source=str(pdf_path),
        template=DatasheetDocument,
        output_dir=str(out_dir),
        backend="llm",                         # Use LLM extraction
        inference="local",                     # Specify local inference execution
        provider_override="ollama",            # Explicitly force Ollama routing
        model_override=GRAPH_EXTRACTION_MODEL,
        use_chunking=True,                     # Keeps context length manageable
        chunk_max_tokens=384,
        processing_mode="many-to-one",
        extraction_contract="dense",
        llm_overrides=LlmRuntimeOverrides(max_output_tokens=4096),
        dense_skeleton_batch_tokens=1536,
        dense_fill_nodes_cap=3,
        debug=True,
    )
    progress_log(f"  Graph: extracting structured entities for {doc_id}", show_progress)
    result = run_pipeline(pipeline_config)
    progress_log(
        f"  Graph: {doc_id} extracted "
        f"({result.knowledge_graph.number_of_nodes()} nodes, "
        f"{result.knowledge_graph.number_of_edges()} edges)",
        show_progress,
    )
    return result.knowledge_graph


def fuse_graphs(graphs: Dict[str, nx.Graph]) -> nx.Graph:
    """Merge per-document graphs into one, using docling-graph's
    deterministic, no-LLM graph fusion (dedup + relationship linking)."""
    converter = GraphConverter()
    fused = converter.fuse(list(graphs.values()))
    return fused


def build_full_graph(
    document_registry: Optional[Dict[str, Path]] = None,
    show_progress: bool = False,
) -> nx.Graph:
    document_registry = document_registry or DOCUMENT_REGISTRY
    per_doc_graphs = {}
    existing_docs = [
        (doc_id, pdf_path)
        for doc_id, pdf_path in document_registry.items()
        if Path(pdf_path).exists()
    ]
    bar = ProgressBar("  Graph documents", len(existing_docs), enabled=show_progress)
    for doc_id, pdf_path in existing_docs:
        if not Path(pdf_path).exists():
            continue
        per_doc_graphs[doc_id] = extract_graph_for_doc(doc_id, str(pdf_path), show_progress=show_progress)
        bar.advance(suffix=doc_id)
    bar.finish()

    if not per_doc_graphs:
        raise RuntimeError(
            "No datasheet PDFs found. Drop documents into data/raw_specs/ and "
            "update DOCUMENT_REGISTRY in config.py."
        )
    if len(per_doc_graphs) == 1:
        return next(iter(per_doc_graphs.values()))
    progress_log("  Graph: fusing document graphs", show_progress)
    return fuse_graphs(per_doc_graphs)


def query_graph_by_entity_name(graph: nx.Graph, entity_name: str) -> List[Dict]:
    """Simple lookup helper: find datasheet graph nodes matching a name
    (case-insensitive substring). Used by tools.py's search_graph tool."""
    entity_name_lower = entity_name.lower()
    hits = []
    for node, attrs in graph.nodes(data=True):
        name = str(attrs.get("name", node)).lower()
        if entity_name_lower in name:
            hits.append({"node": node, "attrs": attrs})
    return hits


def query_graph_neighbors(graph: nx.Graph, node_id: str) -> List[Dict]:
    """Return connected nodes -- e.g. given a register, its bit fields;
    given a peripheral, related pins/registers/timing parameters."""
    if node_id not in graph:
        return []
    return [
        {"node": n, "attrs": graph.nodes[n], "edge": graph.get_edge_data(node_id, n)}
        for n in graph.neighbors(node_id)
    ]
