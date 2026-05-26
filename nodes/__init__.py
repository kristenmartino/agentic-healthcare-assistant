"""LangGraph node implementations.

This package's modules are imported lazily — `nodes.history` pulls FAISS via
`tools.vector_index`, which is a heavy dep we don't want forced on tests that
only touch one node (e.g. tests/test_safety.py).

The public API (the symbols the graph imports) lives in submodules; callers
should `from nodes.X import Y`, not `from nodes import Y`.
"""
