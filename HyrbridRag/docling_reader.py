"""
docling_reader.py

Parses SPI spec PDFs with Docling and produces two categories of chunks:

  1. Structural chunks  -- prose, section by section, token-windowed.
  2. Atomic table chunks -- ONE chunk per register (all its bit fields
     together), never split mid-table. This is what protects bit-level
     retrieval accuracy: a query about "CPHA bit in SPI_CR1" must retrieve
     a chunk that actually contains the full field, not a fragment of a
     table row.

Every chunk carries full provenance metadata (doc_id, page range, section
path) so answers can always cite exact page ranges, matching the
precision-first approach already used in the chip-verification RAG.
"""

from __future__ import annotations
import re
import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Iterable

from docling.document_converter import DocumentConverter
from docling_core.types.doc import DoclingDocument, TableItem, TextItem, SectionHeaderItem

from config import CHUNK_CONFIG


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    chunk_type: str            # "structural" | "atomic_table" | "hierarchical_summary"
    text: str                  # the text actually embedded/retrieved
    page_start: int
    page_end: int
    section_path: str = ""
    # Structured metadata preserved for atomic table chunks (register name,
    # bit range, etc.) -- lets tools filter/answer with exact values
    # without re-parsing prose.
    structured_meta: dict = field(default_factory=dict)

    def to_metadata(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "chunk_type": self.chunk_type,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "section_path": self.section_path,
            **{f"meta_{k}": v for k, v in self.structured_meta.items()},
        }


def _make_chunk_id(doc_id: str, text: str) -> str:
    h = hashlib.sha1(f"{doc_id}:{text}".encode("utf-8")).hexdigest()[:12]
    return f"{doc_id}_{h}"


def parse_document(pdf_path: str) -> DoclingDocument:
    """Run Docling's layout-aware conversion. This is the single source
    of truth for both the vector chunking below and the docling-graph
    extraction in graph_builder.py -- avoids parsing the same PDF twice."""
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    return result.document


def _looks_like_bitfield_table(table: TableItem) -> bool:
    """Heuristic: does this table look like a register/bit-field table?
    Checks header row for common column names. Tune this against your
    actual spec's table headers after a first pass."""
    header_keywords = {
        "bit", "bits", "field", "name", "access", "reset", "type",
        "description", "range", "offset", "address",
    }
    try:
        header_cells = table.export_to_dataframe().columns.tolist()
    except Exception:
        return False
    header_text = " ".join(str(c).lower() for c in header_cells)
    hits = sum(1 for kw in header_keywords if kw in header_text)
    return hits >= 2


def _table_to_atomic_chunks(table: TableItem, doc_id: str, section_path: str) -> List[Chunk]:
    """Convert a bit-field table into one chunk per register.

    Assumes rows are grouped by register (common in SPI specs: a register
    name/address row followed by its bit-field rows). If your spec's
    tables use a different grouping, adjust the grouping logic here --
    this is the single place bit-level chunking quality is decided.
    """
    chunks: List[Chunk] = []
    try:
        df = table.export_to_dataframe()
    except Exception:
        return chunks

    page_no = table.prov[0].page_no if table.prov else 0

    # Group by a "register" column if present, else treat whole table as
    # one chunk (still atomic -- never split mid-table).
    cols_lower = {c: str(c).lower() for c in df.columns}
    register_col = next((c for c, lc in cols_lower.items() if "register" in lc), None)

    if register_col:
        for reg_name, group in df.groupby(register_col):
            text_lines = [f"Register: {reg_name}"]
            for _, row in group.iterrows():
                text_lines.append(" | ".join(f"{c}: {row[c]}" for c in df.columns))
            text = "\n".join(text_lines)
            chunks.append(Chunk(
                chunk_id=_make_chunk_id(doc_id, text),
                doc_id=doc_id,
                chunk_type="atomic_table",
                text=text,
                page_start=page_no,
                page_end=page_no,
                section_path=section_path,
                structured_meta={"register": str(reg_name)},
            ))
    else:
        text = df.to_string(index=False)
        chunks.append(Chunk(
            chunk_id=_make_chunk_id(doc_id, text),
            doc_id=doc_id,
            chunk_type="atomic_table",
            text=text,
            page_start=page_no,
            page_end=page_no,
            section_path=section_path,
        ))
    return chunks


def _split_prose(text: str, max_tokens: int, overlap: int) -> List[str]:
    """Simple whitespace-token windowing with overlap. Swap for a proper
    tokenizer-aware splitter if you want exact token counts matching your
    embedding model's tokenizer."""
    words = text.split()
    if len(words) <= max_tokens:
        return [text]
    chunks = []
    step = max_tokens - overlap
    for i in range(0, len(words), step):
        chunk_words = words[i:i + max_tokens]
        chunks.append(" ".join(chunk_words))
        if i + max_tokens >= len(words):
            break
    return chunks


def build_chunks(doc_id: str, pdf_path: str) -> List[Chunk]:
    """Main entry point: parse a spec PDF and return all chunks
    (structural + atomic table), fully tagged with provenance."""
    doc = parse_document(pdf_path)
    chunks: List[Chunk] = []
    current_section = ""
    prose_buffer: List[str] = []
    prose_page_start: Optional[int] = None

    def flush_prose(page_end: int):
        nonlocal prose_buffer, prose_page_start
        if prose_buffer:
            full_text = " ".join(prose_buffer)
            for piece in _split_prose(
                full_text,
                CHUNK_CONFIG.structural_max_tokens,
                CHUNK_CONFIG.structural_overlap_tokens,
            ):
                chunks.append(Chunk(
                    chunk_id=_make_chunk_id(doc_id, piece),
                    doc_id=doc_id,
                    chunk_type="structural",
                    text=piece,
                    page_start=prose_page_start or page_end,
                    page_end=page_end,
                    section_path=current_section,
                ))
            prose_buffer = []
            prose_page_start = None

    for item, _level in doc.iterate_items():
        page_no = item.prov[0].page_no if getattr(item, "prov", None) else 0

        if isinstance(item, SectionHeaderItem):
            flush_prose(page_no)
            current_section = item.text.strip()

        elif isinstance(item, TableItem):
            flush_prose(page_no)
            if _looks_like_bitfield_table(item):
                chunks.extend(_table_to_atomic_chunks(item, doc_id, current_section))
            else:
                # Non-bit-field tables (e.g. ordering info) still get one
                # atomic chunk each -- tables are never token-windowed.
                try:
                    text = item.export_to_dataframe().to_string(index=False)
                except Exception:
                    text = item.export_to_markdown()
                chunks.append(Chunk(
                    chunk_id=_make_chunk_id(doc_id, text),
                    doc_id=doc_id,
                    chunk_type="atomic_table",
                    text=text,
                    page_start=page_no,
                    page_end=page_no,
                    section_path=current_section,
                ))

        elif isinstance(item, TextItem):
            if prose_page_start is None:
                prose_page_start = page_no
            prose_buffer.append(item.text)

    flush_prose(prose_page_start or 0)
    return chunks
