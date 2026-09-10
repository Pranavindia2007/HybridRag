"""
tools.py

Tool definitions for the agent loop. Same pattern as the chip-verification
RAG: the model decides which tool to call and with what arguments; we
execute and feed results back. Two retrieval tools map directly to the
two halves of the hybrid index (vector store, knowledge graph), plus a
raw page-range fetch for when the model wants to double check context.
"""

from __future__ import annotations
from typing import Dict, List

from document_index import HybridIndex
from graph_builder import query_graph_by_entity_name, query_graph_neighbors
import networkx as nx


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_vector",
            "description": (
                "Hybrid dense+BM25 search over datasheet chunks (prose and "
                "bit-field/register tables, and technical image summaries). "
                "Use for general questions, "
                "conceptual explanations, pin/package lookups, electrical "
                "conditions, or when unsure which datasheet entity a term "
                "belongs to."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "doc_id": {
                        "type": "string",
                        "description": "Optional: restrict to one datasheet doc_id from config.DOCUMENT_REGISTRY",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_graph",
            "description": (
                "Look up a datasheet entity by exact/partial name in the "
                "knowledge graph (e.g. part number, peripheral, pin, register, "
                "bit field, timing parameter, package, or electrical symbol). "
                "Returns the matching node plus structured attributes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_name": {"type": "string"},
                },
                "required": ["entity_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_graph_neighbors",
            "description": (
                "Given a graph node id (from a prior search_graph result), "
                "return connected nodes -- e.g. a register's bit fields, a "
                "peripheral's pins/registers, or a field's related timing "
                "parameters."
            ),
            "parameters": {
                "type": "object",
                "properties": {"node_id": {"type": "string"}},
                "required": ["node_id"],
            },
        },
    },
]


class ToolExecutor:
    def __init__(self, index: HybridIndex, graph: nx.Graph):
        self.index = index
        self.graph = graph

    def execute(self, name: str, arguments: Dict) -> Dict:
        if name == "search_vector":
            hits = self.index.search(
                query=arguments["query"],
                doc_id=arguments.get("doc_id"),
            )
            return {"results": [
                {
                    "chunk_id": h["chunk_id"],
                    "text": h["text"],
                    "doc_id": h["metadata"].get("doc_id"),
                    "page_start": h["metadata"].get("page_start"),
                    "page_end": h["metadata"].get("page_end"),
                    "section": h["metadata"].get("section_path"),
                } for h in hits
            ]}

        elif name == "search_graph":
            entity_name = arguments.get("entity_name")
            if not entity_name:
                return {"error": "Missing required argument: entity_name"}
            hits = query_graph_by_entity_name(self.graph, entity_name)
            return {"results": [{"node_id": h["node"], "attrs": h["attrs"]} for h in hits]}

        elif name == "get_graph_neighbors":
            neighbors = query_graph_neighbors(self.graph, arguments["node_id"])
            return {"results": [
                {"node_id": n["node"], "attrs": n["attrs"], "edge": n["edge"]} for n in neighbors
            ]}

        return {"error": f"Unknown tool: {name}"}
