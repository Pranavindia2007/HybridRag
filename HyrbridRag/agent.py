"""
agent.py

Tool-calling agent loop: multi-iteration, hard-stopped, citation-first.
Same shape as the chip-verification RAG agent -- the model can call
search_vector / search_graph / get_graph_neighbors repeatedly before
answering, but is capped at AGENT_CONFIG.max_tool_iterations and is
instructed to refuse rather than guess when it can't ground an answer
in a retrieved chunk.
"""

from __future__ import annotations
import json
from typing import List, Dict

from config import AGENT_CONFIG
from llm_backend import OllamaBackend
from tools import TOOL_DEFINITIONS, ToolExecutor


SYSTEM_PROMPT = """You are a precision-first assistant for SPI ASIC verification engineers.

Rules:
- Always ground answers in retrieved chunks or graph nodes -- never answer
  from general knowledge about SPI if it might conflict with the specific
  spec(s) loaded.
- Always cite the source: document id + page range (from search_vector)
  or node id (from search_graph) for every factual claim.
- If retrieval doesn't surface a clear answer, say so explicitly rather
  than guessing. Do not fill gaps with assumptions about "typical" SPI
  behavior unless the user asks for general background and you say so.
- For bit-field and register questions, prefer search_graph first (exact
  structured data), then search_vector for any surrounding context/rationale.
- When a question mentions two specs or asks to compare, query both
  doc_ids (or omit doc_id to search across both) and clearly attribute
  each fact to its source spec.
"""


class SPIRagAgent:
    def __init__(self, backend: OllamaBackend, tool_executor: ToolExecutor):
        self.backend = backend
        self.tool_executor = tool_executor
        self.history: List[Dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def ask(self, user_message: str) -> str:
        self.history.append({"role": "user", "content": user_message})

        for _ in range(AGENT_CONFIG.max_tool_iterations):
            message = self.backend.chat(
                messages=self.history,
                tools=TOOL_DEFINITIONS,
                temperature=AGENT_CONFIG.temperature,
            )
            self.history.append(message)

            tool_calls = message.get("tool_calls")
            if not tool_calls:
                return message.get("content", "")

            for call in tool_calls:
                fn = call["function"]
                name = fn["name"]
                args = fn["arguments"]
                if isinstance(args, str):
                    args = json.loads(args)
                result = self.tool_executor.execute(name, args)
                self.history.append({
                    "role": "tool",
                    "content": json.dumps(result),
                })

        # Hard stop reached without a final answer -- surface this plainly
        # rather than silently truncating.
        return (
            "I hit the tool-call iteration limit before reaching a grounded "
            "answer. Try narrowing the question (e.g. name the specific "
            "register or spec)."
        )

    def reset(self):
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
