"""
llm_backend.py

Thin wrapper around the local Ollama server. Keeps chat and embedding
calls in one place so the rest of the code never talks to Ollama's HTTP
API directly -- makes it trivial to swap models or add a remote fallback
later without touching agent.py or document_index.py.
"""

from __future__ import annotations
from typing import List, Dict, Optional
import ollama

from config import OLLAMA_HOST, CHAT_MODEL, CHAT_MODEL_FALLBACK, EMBED_MODEL


class OllamaBackend:
    def __init__(self, host: str = OLLAMA_HOST, chat_model: str = CHAT_MODEL):
        self.client = ollama.Client(host=host)
        self.chat_model = chat_model

    def ensure_models_available(self, models: Optional[List[str]] = None):
        """Sanity check that required models are pulled locally. Raises a
        clear error rather than a confusing runtime failure mid-chat."""
        wanted = models or [self.chat_model, EMBED_MODEL]
        local = {m["model"] for m in self.client.list().get("models", [])}
        missing = [m for m in wanted if m not in local and not any(l.startswith(m) for l in local)]
        if missing:
            raise RuntimeError(
                f"Missing local Ollama models: {missing}. "
                f"Pull them first, e.g.: ollama pull {missing[0]}"
            )

    def chat(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        temperature: float = 0.0,
    ) -> Dict:
        """Single chat turn, with optional tool definitions for tool-calling
        models. Returns the raw Ollama response dict (message, tool_calls)."""
        try:
            response = self.client.chat(
                model=self.chat_model,
                messages=messages,
                tools=tools,
                options={"temperature": temperature},
            )
        except Exception as e:
            if self.chat_model != CHAT_MODEL_FALLBACK:
                # graceful degrade to the smaller model if the primary
                # isn't pulled / OOMs on this GPU
                self.chat_model = CHAT_MODEL_FALLBACK
                response = self.client.chat(
                    model=self.chat_model,
                    messages=messages,
                    tools=tools,
                    options={"temperature": temperature},
                )
            else:
                raise e
        return response["message"]

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Batch embedding call. Ollama's embed endpoint accepts a list
        directly in recent versions; fall back to per-item calls if not."""
        try:
            resp = self.client.embed(model=EMBED_MODEL, input=texts)
            return resp["embeddings"]
        except Exception:
            return [self.client.embeddings(model=EMBED_MODEL, prompt=t)["embedding"] for t in texts]
