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
from graph_builder import query_graph_by_field_name, query_graph_neighbors
import networkx as nx


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_vector",
            "description": (
                "Hybrid dense+BM25 search over spec chunks (prose and "
                "bit-field/register tables). Use for general questions, "
                "conceptual explanations, or when unsure which register a "
                "term belongs to."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "doc_id": {
                        "type": "string",
                        "description": "Optional: restrict to one spec ('spi_spec_a' or 'spi_spec_b')",
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
                "Look up a register or bit-field by exact/partial name in "
                "the knowledge graph (e.g. 'CPHA', 'SPI_CR1'). Returns the "
                "matching node plus its structured attributes (bit range, "
                "access type, reset value). Use when the user names a "
                "specific register or field."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "field_or_register_name": {"type": "string"},
                },
                "required": ["field_or_register_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_graph_neighbors",
            "description": (
                "Given a graph node id (from a prior search_graph result), "
                "return connected nodes -- e.g. a register's bit fields, or "
                "a field's related timing parameters."
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
            hits = query_graph_by_field_name(self.graph, arguments["field_or_register_name"])
            return {"results": [{"node_id": h["node"], "attrs": h["attrs"]} for h in hits]}

        elif name == "get_graph_neighbors":
            neighbors = query_graph_neighbors(self.graph, arguments["node_id"])
            return {"results": [
                {"node_id": n["node"], "attrs": n["attrs"], "edge": n["edge"]} for n in neighbors
            ]}

        return {"error": f"Unknown tool: {name}"}
