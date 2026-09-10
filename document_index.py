"""
document_index.py

Hybrid retrieval index: Chroma for dense/semantic search + BM25 for
exact-match lexical search (part numbers, pin names, register/field
mnemonics, hex addresses -- things dense embeddings routinely blur).

Results from both are combined with reciprocal rank fusion (RRF), which
is simple, has no extra hyperparameters to tune per query, and works well
when you don't have labeled relevance data yet to train a learned reranker.
"""

from __future__ import annotations
import hashlib
from typing import List, Dict, Optional

import chromadb
from rank_bm25 import BM25Okapi

from config import VECTOR_COLLECTION_NAME, VECTOR_STORE_DIR, RETRIEVAL_CONFIG
from docling_reader import Chunk
from llm_backend import OllamaBackend
from progress import ProgressBar, progress_log


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


class HybridIndex:
    def __init__(self, backend: OllamaBackend):
        self.backend = backend
        self.client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
        self.collection = self.client.get_or_create_collection(
            name=VECTOR_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        # BM25 index is rebuilt in-memory from whatever's in Chroma on load
        # (fine at this document scale; move to a persisted index if the
        # corpus grows past a few thousand chunks).
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_chunk_ids: List[str] = []
        self._bm25_docs: List[List[str]] = []

    # ----------------------------------------------------------------
    # Indexing
    # ----------------------------------------------------------------
    def add_chunks(self, chunks: List[Chunk], batch_size: int = 32, show_progress: bool = False):
        if not chunks:
            progress_log("  Vector index: no chunks to add", show_progress)
            return
        self._ensure_unique_ids(chunks, show_progress=show_progress)
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        progress_log(f"  Vector index: embedding/upserting {len(chunks)} chunk(s)", show_progress)
        bar = ProgressBar("  Vector index batches", total_batches, enabled=show_progress)
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c.text for c in batch]
            embeddings = self.backend.embed(texts)
            self.collection.upsert(
                ids=[c.chunk_id for c in batch],
                embeddings=embeddings,
                documents=texts,
                metadatas=[c.to_metadata() for c in batch],
            )
            bar.advance(suffix=f"{min(i + batch_size, len(chunks))}/{len(chunks)} chunks")
        bar.finish()
        progress_log("  Vector index: rebuilding BM25 keyword index", show_progress)
        self._rebuild_bm25()
        progress_log("  Vector index: BM25 ready", show_progress)

    def _ensure_unique_ids(self, chunks: List[Chunk], show_progress: bool = False) -> None:
        seen: set[str] = set()
        repaired = 0
        for index, chunk in enumerate(chunks):
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                continue
            repaired += 1
            base_id = chunk.chunk_id
            salt = hashlib.sha1(
                (
                    f"{base_id}:{index}:{chunk.chunk_type}:"
                    f"{chunk.page_start}:{chunk.page_end}:{chunk.section_path}"
                ).encode("utf-8")
            ).hexdigest()[:8]
            chunk.chunk_id = f"{base_id}_{salt}"
            while chunk.chunk_id in seen:
                salt = hashlib.sha1(f"{chunk.chunk_id}:{repaired}".encode("utf-8")).hexdigest()[:8]
                chunk.chunk_id = f"{base_id}_{salt}"
            seen.add(chunk.chunk_id)
        if repaired:
            progress_log(f"  Vector index: repaired {repaired} duplicate chunk id(s)", show_progress)

    def _rebuild_bm25(self):
        data = self.collection.get(include=["documents"])
        self._bm25_chunk_ids = data["ids"]
        self._bm25_docs = [_tokenize(d) for d in data["documents"]]
        self._bm25 = BM25Okapi(self._bm25_docs) if self._bm25_docs else None

    # ----------------------------------------------------------------
    # Retrieval
    # ----------------------------------------------------------------
    def search(
        self,
        query: str,
        doc_id: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        top_k = top_k or RETRIEVAL_CONFIG.final_top_k
        where = {"doc_id": doc_id} if doc_id else None

        dense_hits = self._dense_search(query, where)
        bm25_hits = self._bm25_search(query, doc_id)

        fused = self._reciprocal_rank_fusion(dense_hits, bm25_hits)
        return fused[:top_k]

    def _dense_search(self, query: str, where: Optional[dict]) -> List[Dict]:
        q_embedding = self.backend.embed([query])[0]
        result = self.collection.query(
            query_embeddings=[q_embedding],
            n_results=RETRIEVAL_CONFIG.dense_top_k,
            where=where,
        )
        hits = []
        for cid, doc, meta, dist in zip(
            result["ids"][0], result["documents"][0],
            result["metadatas"][0], result["distances"][0],
        ):
            hits.append({"chunk_id": cid, "text": doc, "metadata": meta, "score": 1 - dist})
        return hits

    def _bm25_search(self, query: str, doc_id: Optional[str]) -> List[Dict]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(
            zip(self._bm25_chunk_ids, scores, self._bm25_docs),
            key=lambda x: x[1], reverse=True,
        )[:RETRIEVAL_CONFIG.bm25_top_k]

        hits = []
        for cid, score, _ in ranked:
            if score <= 0:
                continue
            data = self.collection.get(ids=[cid], include=["documents", "metadatas"])
            if not data["ids"]:
                continue
            meta = data["metadatas"][0]
            if doc_id and meta.get("doc_id") != doc_id:
                continue
            hits.append({"chunk_id": cid, "text": data["documents"][0], "metadata": meta, "score": score})
        return hits

    def _reciprocal_rank_fusion(self, dense_hits: List[Dict], bm25_hits: List[Dict]) -> List[Dict]:
        k = RETRIEVAL_CONFIG.rrf_k
        scores: Dict[str, float] = {}
        payload: Dict[str, Dict] = {}

        for rank, hit in enumerate(dense_hits):
            scores[hit["chunk_id"]] = scores.get(hit["chunk_id"], 0) + \
                RETRIEVAL_CONFIG.dense_weight / (k + rank + 1)
            payload[hit["chunk_id"]] = hit

        for rank, hit in enumerate(bm25_hits):
            scores[hit["chunk_id"]] = scores.get(hit["chunk_id"], 0) + \
                RETRIEVAL_CONFIG.bm25_weight / (k + rank + 1)
            payload.setdefault(hit["chunk_id"], hit)

        ranked_ids = sorted(scores, key=scores.get, reverse=True)
        return [{**payload[cid], "fused_score": scores[cid]} for cid in ranked_ids]
