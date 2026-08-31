"""
audit_hybrid_quality.py

Offline quality checks for the hybrid RAG ingest:
  1. Ingestion and chunk quality
  2. Graph topology retrievability
  3. Vector-to-graph alignment / hybrid grounding

Usage:
    .venv/bin/python audit_hybrid_quality.py
    .venv/bin/python audit_hybrid_quality.py --terms CPOL CPHA SPIxCON0
    .venv/bin/python audit_hybrid_quality.py --skip-reparse --json data/audit_report.json

The default path avoids LLM calls. Use --live-search to exercise embedding
retrieval through Ollama as an extra check.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import chromadb
import networkx as nx

from config import CHUNK_CONFIG, DOCUMENT_REGISTRY, GRAPH_STORE_DIR, VECTOR_STORE_DIR
from docling_reader import Chunk, build_chunks


DEFAULT_TERMS = [
    "SPI",
    "CPOL",
    "CPHA",
    "SPIxCON0",
    "SPIxCON1",
    "SPIxCON2",
    "SPIxTCNTL",
    "BMODE",
]

REQUIRED_METADATA = {"doc_id", "chunk_type", "page_start", "page_end", "section_path"}
VALID_CHUNK_TYPES = {"structural", "atomic_table", "hierarchical_summary"}
FUSED_GRAPH_PATH = GRAPH_STORE_DIR / "fused_graph.gpickle"


@dataclass
class AuditIssue:
    severity: str
    category: str
    message: str
    details: dict[str, Any]


@dataclass
class AuditReport:
    chunk_summary: dict[str, Any]
    graph_summary: dict[str, Any]
    alignment_summary: dict[str, Any]
    issues: list[AuditIssue]

    @property
    def failed(self) -> bool:
        return any(issue.severity == "ERROR" for issue in self.issues)

    def to_jsonable(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["failed"] = self.failed
        return payload


def _word_count(text: str) -> int:
    return len(text.split())


def _norm(text: Any) -> str:
    return str(text or "").strip().lower()


def _safe_page(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _add_issue(
    issues: list[AuditIssue],
    severity: str,
    category: str,
    message: str,
    **details: Any,
) -> None:
    issues.append(AuditIssue(severity=severity, category=category, message=message, details=details))


def _load_vector_chunks() -> tuple[list[dict[str, Any]], int]:
    client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
    collection = client.get_or_create_collection(
        name="spi_spec_chunks",
        metadata={"hnsw:space": "cosine"},
    )
    count = collection.count()
    data = collection.get(include=["documents", "metadatas"])
    chunks = []
    for cid, doc, meta in zip(data["ids"], data["documents"], data["metadatas"]):
        chunks.append({"chunk_id": cid, "text": doc or "", "metadata": meta or {}})
    return chunks, count


def _load_graph(path: Path) -> nx.Graph | None:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _query_graph_by_field_name(graph: nx.Graph, field_name: str) -> list[dict[str, Any]]:
    field_name_lower = field_name.lower()
    hits = []
    for node, attrs in graph.nodes(data=True):
        name = str(attrs.get("name", node)).lower()
        if field_name_lower in name:
            hits.append({"node": node, "attrs": attrs})
    return hits


def _fresh_chunks_by_doc(skip_reparse: bool) -> dict[str, list[Chunk]]:
    if skip_reparse:
        return {}

    fresh: dict[str, list[Chunk]] = {}
    for doc_id, pdf_path in DOCUMENT_REGISTRY.items():
        if not Path(pdf_path).exists():
            fresh[doc_id] = []
            continue
        fresh[doc_id] = build_chunks(doc_id, str(pdf_path))
    return fresh


def audit_chunk_quality(
    vector_chunks: list[dict[str, Any]],
    vector_count: int,
    fresh_chunks: dict[str, list[Chunk]],
    issues: list[AuditIssue],
) -> dict[str, Any]:
    type_counts = Counter()
    doc_counts = Counter()
    missing_metadata = []
    empty_chunks = []
    invalid_pages = []
    invalid_types = []
    oversized_structural = []
    duplicate_texts = defaultdict(list)

    max_structural_words = int(CHUNK_CONFIG.structural_max_tokens * 1.15)

    for chunk in vector_chunks:
        cid = chunk["chunk_id"]
        text = chunk["text"]
        meta = chunk["metadata"]
        chunk_type = meta.get("chunk_type")
        doc_id = meta.get("doc_id")
        type_counts[chunk_type] += 1
        doc_counts[doc_id] += 1

        missing = sorted(REQUIRED_METADATA - set(meta))
        if missing:
            missing_metadata.append({"chunk_id": cid, "missing": missing})
        if not text.strip():
            empty_chunks.append(cid)
        if chunk_type not in VALID_CHUNK_TYPES:
            invalid_types.append({"chunk_id": cid, "chunk_type": chunk_type})

        page_start = _safe_page(meta.get("page_start"))
        page_end = _safe_page(meta.get("page_end"))
        if page_start is None or page_end is None or page_start < 0 or page_end < page_start:
            invalid_pages.append(
                {"chunk_id": cid, "page_start": meta.get("page_start"), "page_end": meta.get("page_end")}
            )

        if chunk_type == "structural" and _word_count(text) > max_structural_words:
            oversized_structural.append({"chunk_id": cid, "words": _word_count(text)})

        duplicate_texts[_norm(text)].append(cid)

    duplicate_groups = [ids for text, ids in duplicate_texts.items() if text and len(ids) > 1]

    if vector_count == 0:
        _add_issue(issues, "ERROR", "chunk_quality", "Vector store has no chunks.")
    if missing_metadata:
        _add_issue(
            issues,
            "ERROR",
            "chunk_quality",
            "Some chunks are missing required provenance metadata.",
            examples=missing_metadata[:10],
            count=len(missing_metadata),
        )
    if empty_chunks:
        _add_issue(
            issues,
            "ERROR",
            "chunk_quality",
            "Some vector chunks have empty text.",
            examples=empty_chunks[:10],
            count=len(empty_chunks),
        )
    if invalid_pages:
        _add_issue(
            issues,
            "ERROR",
            "chunk_quality",
            "Some chunks have invalid page ranges.",
            examples=invalid_pages[:10],
            count=len(invalid_pages),
        )
    if invalid_types:
        _add_issue(
            issues,
            "ERROR",
            "chunk_quality",
            "Some chunks use unknown chunk_type values.",
            examples=invalid_types[:10],
            count=len(invalid_types),
        )
    if oversized_structural:
        _add_issue(
            issues,
            "WARN",
            "chunk_quality",
            "Some structural chunks exceed the configured token window by more than 15%.",
            examples=oversized_structural[:10],
            count=len(oversized_structural),
            configured_max_tokens=CHUNK_CONFIG.structural_max_tokens,
        )
    if duplicate_groups:
        _add_issue(
            issues,
            "WARN",
            "chunk_quality",
            "Duplicate chunk texts were found in the vector store.",
            examples=duplicate_groups[:5],
            count=len(duplicate_groups),
        )
    if vector_count and type_counts.get("atomic_table", 0) == 0:
        _add_issue(
            issues,
            "WARN",
            "chunk_quality",
            "No atomic_table chunks were found; register/table lookup may be fragile.",
        )

    fresh_summary = {}
    if fresh_chunks:
        vector_ids = {chunk["chunk_id"] for chunk in vector_chunks}
        for doc_id, chunks in fresh_chunks.items():
            fresh_ids = {chunk.chunk_id for chunk in chunks}
            missing_from_vector = sorted(fresh_ids - vector_ids)
            stale_in_vector = sorted(
                chunk["chunk_id"]
                for chunk in vector_chunks
                if chunk["metadata"].get("doc_id") == doc_id and chunk["chunk_id"] not in fresh_ids
            )
            fresh_summary[doc_id] = {
                "fresh_chunk_count": len(chunks),
                "fresh_type_counts": dict(Counter(chunk.chunk_type for chunk in chunks)),
                "missing_from_vector": len(missing_from_vector),
                "stale_in_vector": len(stale_in_vector),
            }
            if missing_from_vector or stale_in_vector:
                _add_issue(
                    issues,
                    "WARN",
                    "chunk_quality",
                    "Persisted vector chunks differ from freshly rebuilt chunks.",
                    doc_id=doc_id,
                    missing_examples=missing_from_vector[:10],
                    stale_examples=stale_in_vector[:10],
                    missing_count=len(missing_from_vector),
                    stale_count=len(stale_in_vector),
                )

    return {
        "vector_count": vector_count,
        "loaded_vector_chunks": len(vector_chunks),
        "chunk_type_counts": dict(type_counts),
        "doc_counts": dict(doc_counts),
        "fresh_ingestion": fresh_summary,
    }


def _graph_term(attrs: dict[str, Any], node_id: Any) -> str:
    name = attrs.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return str(node_id)


def _extract_graph_terms(graph: nx.Graph, max_terms: int) -> list[str]:
    terms = []
    seen = set()
    for node, attrs in graph.nodes(data=True):
        term = str(attrs.get("name") or "").strip()
        if not (2 <= len(term) <= 80):
            continue
        if _norm(term) in seen:
            continue
        seen.add(_norm(term))
        terms.append(term)
        if len(terms) >= max_terms:
            break
    return terms


def audit_graph_topology(graph: nx.Graph | None, terms: list[str], issues: list[AuditIssue]) -> dict[str, Any]:
    if graph is None:
        _add_issue(issues, "ERROR", "graph_topology", f"Graph file not found: {FUSED_GRAPH_PATH}")
        return {"exists": False}

    label_counts = Counter(attrs.get("label") or attrs.get("__class__") or "unknown" for _, attrs in graph.nodes(data=True))
    isolated = list(nx.isolates(graph))
    named_isolated = [
        {"node_id": node, "name": _graph_term(graph.nodes[node], node)}
        for node in isolated
        if _graph_term(graph.nodes[node], node)
    ]

    if graph.number_of_nodes() == 0:
        _add_issue(issues, "ERROR", "graph_topology", "Knowledge graph has zero nodes.")
    if graph.number_of_edges() == 0 and graph.number_of_nodes() > 1:
        _add_issue(issues, "ERROR", "graph_topology", "Knowledge graph has nodes but no edges.")
    if named_isolated:
        _add_issue(
            issues,
            "WARN",
            "graph_topology",
            "Some named graph nodes are isolated and may not support neighborhood retrieval.",
            examples=named_isolated[:10],
            count=len(named_isolated),
        )

    term_checks = {}
    for term in terms:
        hits = _query_graph_by_field_name(graph, term)
        degrees = [int(graph.degree(hit["node"])) for hit in hits if hit["node"] in graph]
        term_checks[term] = {"hits": len(hits), "max_degree": max(degrees) if degrees else 0}
        if not hits:
            _add_issue(
                issues,
                "WARN",
                "graph_topology",
                "Expected retrieval term was not found in the graph.",
                term=term,
            )
        elif max(degrees or [0]) == 0:
            _add_issue(
                issues,
                "WARN",
                "graph_topology",
                "Graph term resolves only to isolated nodes.",
                term=term,
                hits=len(hits),
            )

    if graph.is_directed():
        component_count = nx.number_weakly_connected_components(graph)
    else:
        component_count = nx.number_connected_components(graph)

    return {
        "exists": True,
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "is_directed": graph.is_directed(),
        "component_count": component_count,
        "isolated_nodes": len(isolated),
        "label_counts": dict(label_counts),
        "term_checks": term_checks,
    }


def _contains_term(text: str, term: str) -> bool:
    escaped = re.escape(term.strip())
    if not escaped:
        return False
    return re.search(rf"(?<!\w){escaped}(?!\w)", text, flags=re.IGNORECASE) is not None


def audit_vector_graph_alignment(
    vector_chunks: list[dict[str, Any]],
    graph: nx.Graph | None,
    terms: list[str],
    max_graph_terms: int,
    issues: list[AuditIssue],
) -> dict[str, Any]:
    if graph is None:
        return {"checked_terms": 0, "aligned_terms": 0, "missing_terms": []}

    graph_terms = _extract_graph_terms(graph, max_graph_terms)
    checked_terms = []
    seen = set()
    for term in [*terms, *graph_terms]:
        key = _norm(term)
        if key and key not in seen:
            seen.add(key)
            checked_terms.append(term)

    missing_terms = []
    aligned_terms = {}
    for term in checked_terms:
        matches = []
        for chunk in vector_chunks:
            if _contains_term(chunk["text"], term):
                meta = chunk["metadata"]
                matches.append(
                    {
                        "chunk_id": chunk["chunk_id"],
                        "doc_id": meta.get("doc_id"),
                        "page_start": meta.get("page_start"),
                        "page_end": meta.get("page_end"),
                        "chunk_type": meta.get("chunk_type"),
                    }
                )
        if matches:
            aligned_terms[term] = matches[:5]
        else:
            missing_terms.append(term)

    if missing_terms:
        _add_issue(
            issues,
            "WARN",
            "hybrid_alignment",
            "Some graph terms were not found verbatim in vector chunks.",
            examples=missing_terms[:20],
            count=len(missing_terms),
        )

    return {
        "checked_terms": len(checked_terms),
        "aligned_terms": len(aligned_terms),
        "missing_terms": missing_terms,
        "sample_alignments": dict(list(aligned_terms.items())[:10]),
    }


def audit_live_search(terms: Iterable[str], issues: list[AuditIssue]) -> dict[str, Any]:
    try:
        from document_index import HybridIndex
        from llm_backend import OllamaBackend

        index = HybridIndex(OllamaBackend())
        checks = {}
        for term in terms:
            hits = index.search(term, top_k=3)
            checks[term] = [
                {
                    "chunk_id": hit["chunk_id"],
                    "score": hit.get("score"),
                    "fused_score": hit.get("fused_score"),
                    "page_start": hit["metadata"].get("page_start"),
                    "page_end": hit["metadata"].get("page_end"),
                    "chunk_type": hit["metadata"].get("chunk_type"),
                }
                for hit in hits
            ]
            if not hits:
                _add_issue(
                    issues,
                    "WARN",
                    "live_search",
                    "Live vector search returned no hits for term.",
                    term=term,
                )
        return {"enabled": True, "checks": checks}
    except Exception as exc:
        _add_issue(
            issues,
            "WARN",
            "live_search",
            "Live vector search could not run.",
            error=str(exc),
        )
        return {"enabled": True, "error": str(exc)}


def print_report(report: AuditReport) -> None:
    severities = Counter(issue.severity for issue in report.issues)
    print("\n== Hybrid RAG Quality Audit ==")
    print(f"Status: {'FAIL' if report.failed else 'PASS'}")
    print(f"Issues: {dict(severities)}")

    print("\n-- Chunk Quality --")
    print(json.dumps(report.chunk_summary, indent=2, sort_keys=True))

    print("\n-- Graph Topology --")
    print(json.dumps(report.graph_summary, indent=2, sort_keys=True))

    print("\n-- Vector-to-Graph Alignment --")
    alignment = dict(report.alignment_summary)
    if "missing_terms" in alignment and len(alignment["missing_terms"]) > 20:
        alignment["missing_terms"] = alignment["missing_terms"][:20] + ["..."]
    print(json.dumps(alignment, indent=2, sort_keys=True))

    if report.issues:
        print("\n-- Issues --")
        for issue in report.issues:
            print(f"[{issue.severity}] {issue.category}: {issue.message}")
            if issue.details:
                print(f"  {json.dumps(issue.details, sort_keys=True)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit hybrid RAG ingestion, graph, and grounding quality.")
    parser.add_argument("--skip-reparse", action="store_true", help="Do not rebuild chunks from PDFs for comparison.")
    parser.add_argument("--live-search", action="store_true", help="Run live HybridIndex.search checks through Ollama.")
    parser.add_argument("--json", type=Path, help="Optional path to write the full audit report as JSON.")
    parser.add_argument("--max-graph-terms", type=int, default=40, help="Max graph node terms to align against vectors.")
    parser.add_argument("--terms", nargs="*", default=DEFAULT_TERMS, help="Specific terms expected to be retrievable.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    issues: list[AuditIssue] = []

    vector_chunks, vector_count = _load_vector_chunks()
    fresh_chunks = _fresh_chunks_by_doc(args.skip_reparse)
    graph = _load_graph(FUSED_GRAPH_PATH)

    chunk_summary = audit_chunk_quality(vector_chunks, vector_count, fresh_chunks, issues)
    graph_summary = audit_graph_topology(graph, args.terms, issues)
    alignment_summary = audit_vector_graph_alignment(
        vector_chunks,
        graph,
        args.terms,
        args.max_graph_terms,
        issues,
    )
    if args.live_search:
        alignment_summary["live_search"] = audit_live_search(args.terms, issues)

    report = AuditReport(
        chunk_summary=chunk_summary,
        graph_summary=graph_summary,
        alignment_summary=alignment_summary,
        issues=issues,
    )
    print_report(report)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report.to_jsonable(), indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nWrote JSON report: {args.json}")

    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
