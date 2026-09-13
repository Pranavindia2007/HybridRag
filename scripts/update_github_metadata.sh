#!/usr/bin/env bash
set -euo pipefail

repo="Pranavindia2007/HybridRag"
description="Offline hybrid RAG for semiconductor datasheets with Docling, ChromaDB, BM25, Ollama vision summaries, and a knowledge graph."
homepage="https://pranavindia2007.github.io/HybridRag/"

topics=(
  rag
  hybrid-rag
  offline-rag
  semiconductor
  datasheet
  datasheet-rag
  graph-rag
  knowledge-graph
  vector-search
  bm25
  chromadb
  ollama
  docling
  docling-graph
  technical-pdf
  asic-verification
  soc-verification
  embedded-systems
  retrieval-augmented-generation
  ai-engineering
)

gh repo edit "$repo" \
  --description "$description" \
  --homepage "$homepage"

args=(--method PUT -H "Accept: application/vnd.github+json" "/repos/$repo/topics")
for topic in "${topics[@]}"; do
  args+=(-f "names[]=$topic")
done

gh api "${args[@]}"

echo "Updated $repo description, homepage, and topics."
