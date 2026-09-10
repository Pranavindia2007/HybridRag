"""
docling_reader.py

Parses semiconductor datasheet PDFs with Docling and produces three
categories of chunks:

  1. Structural chunks  -- prose, section by section, token-windowed.
  2. Atomic table chunks -- ONE chunk per register (all its bit fields
     together), never split mid-table. This is what protects bit-level
     retrieval accuracy: a query about a field in a control register must retrieve
     a chunk that actually contains the full field, not a fragment of a
     table row.
  3. Image analysis chunks -- vision-model summaries for technical visuals
     such as timing diagrams, flowcharts, and block diagrams.

Every chunk carries full provenance metadata (doc_id, page range, section
path) so answers can always cite exact page ranges, matching the
precision-first approach already used in the chip-verification RAG.
"""

from __future__ import annotations
import json
import re
import hashlib
from dataclasses import dataclass, field
from typing import Any, List, Optional, TYPE_CHECKING

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import PdfFormatOption
from docling_core.types.doc import DoclingDocument, PictureItem, TableItem, TextItem, SectionHeaderItem

from config import CHUNK_CONFIG, IMAGE_PROCESSING_CONFIG
from progress import ProgressBar, progress_log

if TYPE_CHECKING:
    from llm_backend import OllamaBackend


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    chunk_type: str            # "structural" | "atomic_table" | "image_analysis" | "hierarchical_summary"
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
            **{f"meta_{k}": _metadata_value(v) for k, v in self.structured_meta.items()},
        }


def _make_chunk_id(doc_id: str, text: str, salt: str = "") -> str:
    h = hashlib.sha1(f"{doc_id}:{salt}:{text}".encode("utf-8")).hexdigest()[:12]
    return f"{doc_id}_{h}"


def _metadata_value(value: Any) -> str | int | float | bool:
    if isinstance(value, (str, int, float, bool)):
        return value
    if value is None:
        return ""
    return json.dumps(value, sort_keys=True)


def _assign_unique_chunk_ids(chunks: List[Chunk], show_progress: bool = False) -> None:
    """Make chunk ids deterministic and unique within a document ingest.

    Repeated headers, boilerplate, and identical short tables can produce the
    same text on multiple pages. Chroma allows upserts to existing ids, but not
    duplicate ids in a single request, and repeated text still needs distinct
    provenance. The salt keeps those chunks separately addressable.
    """
    signature_counts: dict[str, int] = {}
    used_ids: set[str] = set()
    duplicate_count = 0
    repeated_signature_count = 0

    for chunk in chunks:
        signature = json.dumps(
            {
                "chunk_type": chunk.chunk_type,
                "text": chunk.text,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "section_path": chunk.section_path,
                "structured_meta": chunk.structured_meta,
            },
            default=str,
            sort_keys=True,
        )
        occurrence = signature_counts.get(signature, 0)
        signature_counts[signature] = occurrence + 1
        if occurrence:
            repeated_signature_count += 1

        salt = json.dumps(
            {
                "chunk_type": chunk.chunk_type,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "section_path": chunk.section_path,
                "occurrence": occurrence,
            },
            sort_keys=True,
        )
        chunk.chunk_id = _make_chunk_id(chunk.doc_id, chunk.text, salt=salt)
        while chunk.chunk_id in used_ids:
            duplicate_count += 1
            occurrence += 1
            salt = f"{salt}:{occurrence}"
            chunk.chunk_id = _make_chunk_id(chunk.doc_id, chunk.text, salt=salt)
        used_ids.add(chunk.chunk_id)

    if repeated_signature_count or duplicate_count:
        progress_log(
            f"  Chunking: assigned unique IDs "
            f"({repeated_signature_count} repeated chunk signature(s), "
            f"{duplicate_count} hash collision(s))",
            show_progress,
        )


def parse_document(pdf_path: str, generate_images: bool = False) -> DoclingDocument:
    """Run Docling's layout-aware conversion. This is the single source
    of truth for both the vector chunking below and the docling-graph
    extraction in graph_builder.py -- avoids parsing the same PDF twice."""
    if generate_images:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.images_scale = IMAGE_PROCESSING_CONFIG.images_scale
        pipeline_options.generate_picture_images = True
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
    else:
        converter = DocumentConverter()
    result = converter.convert(pdf_path)
    return result.document


def _looks_like_bitfield_table(table: TableItem) -> bool:
    """Heuristic: does this table look like a register/bit-field table?
    Checks header row for common column names. Tune this against your
    actual datasheet's table headers after a first pass."""
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

    Assumes rows are grouped by register (common in chip datasheets: a
    register name/address row followed by its bit-field rows). If your datasheet's
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


def build_chunks(
    doc_id: str,
    pdf_path: str,
    backend: Optional["OllamaBackend"] = None,
    process_images: bool = False,
    analyze_uncached_images: bool = True,
    show_progress: bool = False,
) -> List[Chunk]:
    """Main entry point: parse a datasheet PDF and return all chunks
    (structural + atomic table + image analysis), fully tagged with provenance."""
    progress_log(f"  Docling: converting {pdf_path}", show_progress)
    doc = parse_document(pdf_path, generate_images=process_images)
    progress_log("  Docling: conversion complete", show_progress)

    chunks: List[Chunk] = []
    current_section = ""
    prose_buffer: List[str] = []
    prose_page_start: Optional[int] = None
    image_candidates = []
    picture_counter = 0
    item_count = 0
    table_count = 0
    text_count = 0
    section_count = 0

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

    items = list(doc.iterate_items())
    item_bar = ProgressBar("  Chunking document items", len(items), enabled=show_progress)
    for item, _level in items:
        item_count += 1
        page_no = item.prov[0].page_no if getattr(item, "prov", None) else 0

        if isinstance(item, SectionHeaderItem):
            section_count += 1
            flush_prose(page_no)
            current_section = item.text.strip()

        elif isinstance(item, TableItem):
            table_count += 1
            flush_prose(page_no)
            if _looks_like_bitfield_table(item):
                chunks.extend(_table_to_atomic_chunks(item, doc_id, current_section))
            else:
                # Non-bit-field tables (e.g. ordering, pinout, electrical
                # characteristics) still get one
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

        elif isinstance(item, PictureItem):
            flush_prose(page_no)
            if process_images:
                try:
                    image = item.get_image(doc)
                except Exception:
                    image = None
                if image is not None:
                    from image_processor import ImageCandidate

                    picture_counter += 1
                    image_candidates.append(ImageCandidate(
                        doc_id=doc_id,
                        page_no=page_no,
                        section_path=current_section,
                        image=image,
                        ordinal=picture_counter,
                    ))

        elif isinstance(item, TextItem):
            text_count += 1
            if prose_page_start is None:
                prose_page_start = page_no
            prose_buffer.append(item.text)

        item_bar.advance(suffix=f"page {page_no}")

    item_bar.finish()
    flush_prose(prose_page_start or 0)
    progress_log(
        f"  Chunking: scanned {item_count} item(s), "
        f"{section_count} section header(s), {text_count} text item(s), "
        f"{table_count} table(s), {len(image_candidates)} image candidate(s)",
        show_progress,
    )
    if process_images and image_candidates:
        if backend is None:
            from llm_backend import OllamaBackend

            backend = OllamaBackend()
        from image_processor import ImageProcessor

        image_processor = ImageProcessor(backend)
        image_results = image_processor.process_all_with_progress(
            image_candidates,
            show_progress=show_progress,
            analyze_uncached=analyze_uncached_images,
        )
        indexed_images = set()
        for result in image_results:
            if not result.should_index:
                continue
            if (
                not IMAGE_PROCESSING_CONFIG.index_duplicate_occurrences
                and result.image_sha256 in indexed_images
            ):
                continue
            indexed_images.add(result.image_sha256)
            text = result.to_chunk_text()
            chunks.append(Chunk(
                chunk_id=_make_chunk_id(doc_id, text),
                doc_id=doc_id,
                chunk_type="image_analysis",
                text=text,
                page_start=result.page_no,
                page_end=result.page_no,
                section_path=result.section_path,
                structured_meta=result.to_structured_meta(),
            ))
    _assign_unique_chunk_ids(chunks, show_progress=show_progress)

    if show_progress:
        atomic = sum(1 for c in chunks if c.chunk_type == "atomic_table")
        structural = sum(1 for c in chunks if c.chunk_type == "structural")
        images = sum(1 for c in chunks if c.chunk_type == "image_analysis")
        progress_log(
            f"  Chunking complete: {len(chunks)} total "
            f"({structural} structural, {atomic} table, {images} image)",
        )
    return chunks
