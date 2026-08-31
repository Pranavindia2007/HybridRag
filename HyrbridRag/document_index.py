"""
document_index.py

Hybrid retrieval index: Chroma for dense/semantic search + BM25 for
exact-match lexical search (register names, bit-field mnemonics like
CPOL/CPHA, hex addresses -- things dense embeddings routinely blur).

Results from both are combined with reciprocal rank fusion (RRF), which
is simple, has no extra hyperparameters to tune per query, and works well
when you don't have labeled relevance data yet to train a learned reranker.
"""

from __future__ import annotations
from typing import List, Dict, Optional
from dataclasses import asdict

import chromadb
from rank_bm25 import BM25Okapi

from config import VECTOR_STORE_DIR, RETRIEVAL_CONFIG
from docling_reader import Chunk
from llm_backend import OllamaBackend


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


class HybridIndex:
    def __init__(self, backend: OllamaBackend):
        self.backend = backend
        self.client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
        self.collection = self.client.get_or_create_collection(
            name="spi_spec_chunks",
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
    def add_chunks(self, chunks: List[Chunk], batch_size: int = 32):
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
        self._rebuild_bm25()

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
