"""
config.py
Central configuration for the semiconductor datasheet RAG system.
All paths, model names, and tunables live here so the rest of the
codebase never hardcodes them.
"""

from pathlib import Path
from dataclasses import dataclass


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DOCS_DIR = DATA_DIR / "raw_specs"          # drop your input PDFs here
VECTOR_STORE_DIR = DATA_DIR / "chroma_store"    # Chroma persistence
GRAPH_STORE_DIR = DATA_DIR / "graph_store"      # docling-graph outputs
CACHE_DIR = DATA_DIR / "cache"                  # parsed-doc / chunk cache
IMAGE_CACHE_DIR = CACHE_DIR / "images"          # extracted/cached picture crops
IMAGE_CACHE_PATH = CACHE_DIR / "image_cache.json"

for d in (DATA_DIR, RAW_DOCS_DIR, VECTOR_STORE_DIR, GRAPH_STORE_DIR, CACHE_DIR, IMAGE_CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Document registry
# --------------------------------------------------------------------------
# Add one entry per datasheet. doc_id is the namespace used everywhere
# (chunk metadata, vector filters, graph fusion, citations).
DOCUMENT_REGISTRY = {
    "sak_tc389qp_160f400s_ae": RAW_DOCS_DIR / "infineon-tc38x-datasheet-en.pdf",
    # "another_chip": RAW_DOCS_DIR / "another-chip-datasheet.pdf",
}

VECTOR_COLLECTION_NAME = "datasheet_chunks"


# --------------------------------------------------------------------------
# Ollama models (all local / offline)
# --------------------------------------------------------------------------
OLLAMA_HOST = "http://localhost:11434"

# Chat / generation model (SLM). Swap for whatever fits your VRAM budget.
CHAT_MODEL = "gpt-oss:20b"       # good accuracy/VRAM tradeoff on consumer GPUs
CHAT_MODEL_FALLBACK = "phi4"             # smaller fallback if VRAM constrained

# Embedding model for the vector store.
EMBED_MODEL = "nomic-embed-text"         # 768-dim, cheap, strong for technical text
EMBED_DIM = 768

# LLM used by docling-graph for structured extraction (kept small on purpose;
# extraction is schema-constrained, so a big model isn't necessary).
GRAPH_EXTRACTION_MODEL = "qwen2.5-coder:7b"

# Vision model used only during ingestion to describe meaningful diagrams,
# charts, timing waveforms, and flowcharts found as document images.
VISION_MODEL = "qwen2.5vl"


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
@dataclass
class ChunkConfig:
    # Structural (prose) chunking
    structural_max_tokens: int = 400
    structural_overlap_tokens: int = 50

    # Atomic table chunking (bit-field / register tables)
    # One chunk per logical row group (e.g. one register's full bit map),
    # never split mid-row. This is the setting that protects bit-level
    # accuracy -- do not raise structural chunking to swallow tables.
    atomic_table_max_rows_per_chunk: int = 1  # 1 = one register per chunk

    # RAPTOR-style hierarchical summary chunks (optional, built on top of
    # structural chunks) -- helps with "what does this whole section cover"
    # style questions without hurting bit-level precision, since summaries
    # are additive, not a replacement for atomic chunks.
    enable_hierarchical_summaries: bool = True


CHUNK_CONFIG = ChunkConfig()


@dataclass
class ImageProcessingConfig:
    enabled: bool = True
    vision_model: str = VISION_MODEL
    max_workers: int = 2
    images_scale: float = 2.0
    cache_path: Path = IMAGE_CACHE_PATH
    image_dir: Path = IMAGE_CACHE_DIR
    perceptual_hash_hamming_threshold: int = 3
    min_width: int = 32
    min_height: int = 32
    index_duplicate_occurrences: bool = False


IMAGE_PROCESSING_CONFIG = ImageProcessingConfig()

# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------
@dataclass
class RetrievalConfig:
    dense_top_k: int = 8
    bm25_top_k: int = 8
    # Final fused/reranked result count handed to the LLM
    final_top_k: int = 6
    # Weight for reciprocal-rank fusion between dense and BM25
    rrf_k: int = 60
    # Part numbers, pins, fields/registers, addresses, and symbols are
    # exact-match sensitive (SAK-TC389QP, P20.3, ENDINIT, 0xF0036000...) --
    # BM25 tends to win here, dense wins on paraphrased conceptual queries.
    bm25_weight: float = 0.5
    dense_weight: float = 0.5


RETRIEVAL_CONFIG = RetrievalConfig()


# --------------------------------------------------------------------------
# Agent loop
# --------------------------------------------------------------------------
@dataclass
class AgentConfig:
    max_tool_iterations: int = 6   # hard stop to prevent runaway loops
    temperature: float = 0.0       # deterministic, precision-first
    require_citation: bool = True  # refuse to answer without a grounded chunk


AGENT_CONFIG = AgentConfig()
