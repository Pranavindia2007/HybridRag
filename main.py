"""
main.py

Simple CLI chatbot for querying semiconductor datasheet(s). Run `python ingest.py`
first to build the indices. This is intentionally minimal -- the test
scenario generation tool will build on top of DatasheetRagAgent + the same
hybrid index rather than replacing them.

Usage:
    python main.py
"""

from __future__ import annotations
import pickle
import sys

from config import AGENT_CONFIG
from llm_backend import OllamaBackend
from document_index import HybridIndex
from graph_builder import build_full_graph
from tools import ToolExecutor
from agent import DatasheetRagAgent
from ingest import FUSED_GRAPH_PATH


def load_graph():
    if FUSED_GRAPH_PATH.exists():
        with open(FUSED_GRAPH_PATH, "rb") as f:
            return pickle.load(f)
    print("No cached graph found -- building now (this calls the local LLM "
          "for extraction and may take a while)...")
    return build_full_graph()


def main():
    backend = OllamaBackend()
    try:
        backend.ensure_models_available()
    except RuntimeError as e:
        print(f"Setup issue: {e}")
        sys.exit(1)

    index = HybridIndex(backend)
    graph = load_graph()
    executor = ToolExecutor(index, graph)
    agent = DatasheetRagAgent(backend, executor)

    print("Datasheet Chatbot -- offline, grounded in your loaded datasheet(s).")
    print("Type 'exit' to quit, 'reset' to clear conversation history.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break
        if user_input.lower() == "reset":
            agent.reset()
            print("(conversation reset)\n")
            continue

        answer = agent.ask(user_input)
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    main()
