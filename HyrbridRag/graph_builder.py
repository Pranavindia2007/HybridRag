"""
graph_builder.py

Wraps docling-graph to build the structured/relational half of the hybrid
index. Runs the SPISpecDocument Pydantic template (schemas/spi_ontology.py)
against each spec, then uses docling-graph's built-in fusion to merge both
specs into one cross-document graph -- e.g. so "does spec A's default CPOL
match spec B's" becomes a graph traversal instead of two separate lookups.

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
from schemas.spi_ontology import SPISpecDocument


def extract_graph_for_doc(doc_id: str, pdf_path: str) -> nx.Graph:
    """Run the docling-graph pipeline for a single spec document."""
    out_dir = GRAPH_STORE_DIR / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline_config = PipelineConfig(
        source=str(pdf_path),
        template=SPISpecDocument,
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
    result = run_pipeline(pipeline_config)
    return result.knowledge_graph


def fuse_graphs(graphs: Dict[str, nx.Graph]) -> nx.Graph:
    """Merge per-document graphs into one, using docling-graph's
    deterministic, no-LLM graph fusion (dedup + relationship linking)."""
    converter = GraphConverter()
    fused = converter.fuse(list(graphs.values()))
    return fused


def build_full_graph(document_registry: Optional[Dict[str, Path]] = None) -> nx.Graph:
    document_registry = document_registry or DOCUMENT_REGISTRY
    per_doc_graphs = {}
    for doc_id, pdf_path in document_registry.items():
        if not Path(pdf_path).exists():
            continue
        per_doc_graphs[doc_id] = extract_graph_for_doc(doc_id, str(pdf_path))

    if not per_doc_graphs:
        raise RuntimeError(
            "No spec PDFs found. Drop your specs into data/raw_specs/ and "
            "update DOCUMENT_REGISTRY in config.py."
        )
    if len(per_doc_graphs) == 1:
        return next(iter(per_doc_graphs.values()))
    return fuse_graphs(per_doc_graphs)


def query_graph_by_field_name(graph: nx.Graph, field_name: str) -> List[Dict]:
    """Simple lookup helper: find bit-field/register nodes matching a name
    (case-insensitive substring). Used by tools.py's search_graph tool."""
    field_name_lower = field_name.lower()
    hits = []
    for node, attrs in graph.nodes(data=True):
        name = str(attrs.get("name", node)).lower()
        if field_name_lower in name:
            hits.append({"node": node, "attrs": attrs})
    return hits


def query_graph_neighbors(graph: nx.Graph, node_id: str) -> List[Dict]:
    """Return connected nodes -- e.g. given a register, its bit fields;
    given a bit field, its parent register and any timing dependency."""
    if node_id not in graph:
        return []
    return [
        {"node": n, "attrs": graph.nodes[n], "edge": graph.get_edge_data(node_id, n)}
        for n in graph.neighbors(node_id)
    ]
