"""Tests for the multi-intent sub-query extractor in nodes.medical_search_node.

The extractor's job is to take a bundled user message like:
    "My father has CKD. Book a nephrologist for him and summarize the latest
     treatment methods."
…and turn it into something a site-restricted MedlinePlus/WHO query can
actually find ("latest treatments for chronic kidney disease"). When the
search branch is the only intent, the user input passes through unchanged.

We test the heuristic + the LLM-driven path separately so behavior is
verifiable even when no API key is configured.
"""
from __future__ import annotations

from nodes.medical_search_node import (
    _extract_search_subquery,
    _heuristic_search_query,
)


def test_passthrough_single_intent():
    """No other intent → query is sent verbatim."""
    state = {"user_input": "what are the symptoms of pneumonia",
             "intents": ["medical_search"]}
    refined, method = _extract_search_subquery(state)
    assert method == "passthrough"
    assert refined == "what are the symptoms of pneumonia"


def test_heuristic_strips_book_clause():
    out = _heuristic_search_query(
        "Book a nephrologist for him and summarize the latest treatment methods"
    )
    assert "Book" not in out and "book" not in out
    # leading conjunction also stripped
    assert "and " not in out.lower().split(".", 1)[0]
    assert "treatment methods" in out.lower()


def test_heuristic_strips_schedule_clause():
    out = _heuristic_search_query(
        "Schedule me a cardiologist next week and what is afib"
    )
    assert "schedule" not in out.lower()
    assert "afib" in out.lower()


def test_heuristic_preserves_topic_sentence():
    """Topic-only input should be returned essentially unchanged."""
    text = "latest treatments for parkinson's"
    assert _heuristic_search_query(text) == text


def test_heuristic_falls_back_to_input_when_strip_empties_it():
    """If the booking pattern would erase everything, return the original
    so downstream search at least has *something* to look up."""
    text = "book me an appointment"
    out = _heuristic_search_query(text)
    assert out  # never empty


def test_extractor_uses_heuristic_when_llm_returns_garbage(monkeypatch):
    """The extractor should fall through to the heuristic when the LLM
    returns something nonsensical (out-of-range length, or the input echo)."""
    monkeypatch.setattr(
        "nodes.medical_search_node.chat",
        lambda **kwargs: "",  # empty string fails the sanity floor
    )
    state = {
        "user_input": "My father has CKD. Book a nephrologist and tell me about treatments.",
        "intents": ["booking", "medical_search"],
    }
    refined, method = _extract_search_subquery(state)
    assert method == "heuristic"
    assert "book" not in refined.lower()


def test_extractor_uses_llm_when_available(monkeypatch):
    monkeypatch.setattr(
        "nodes.medical_search_node.chat",
        lambda **kwargs: "latest treatment for chronic kidney disease",
    )
    state = {
        "user_input": "My 70-year-old father has chronic kidney disease. "
                      "Book a nephrologist for him and summarize the latest treatment methods.",
        "intents": ["booking", "medical_search"],
    }
    refined, method = _extract_search_subquery(state)
    assert method == "llm"
    assert refined == "latest treatment for chronic kidney disease"


def test_extractor_strips_surrounding_quotes_from_llm(monkeypatch):
    """LLMs sometimes wrap the answer in quotes. Trim them before use."""
    monkeypatch.setattr(
        "nodes.medical_search_node.chat",
        lambda **kwargs: '"symptoms of pneumonia"',
    )
    state = {
        "user_input": "Book a pulmonologist and what are the symptoms of pneumonia",
        "intents": ["booking", "medical_search"],
    }
    refined, method = _extract_search_subquery(state)
    assert method == "llm"
    assert refined == "symptoms of pneumonia"


def test_extractor_rejects_overly_long_llm_output(monkeypatch):
    """If the LLM produces a giant blob, treat it as garbage and fall back."""
    long_output = "x " * 200
    monkeypatch.setattr(
        "nodes.medical_search_node.chat",
        lambda **kwargs: long_output,
    )
    state = {
        "user_input": "Book a nephrologist and tell me about CKD treatments",
        "intents": ["booking", "medical_search"],
    }
    refined, method = _extract_search_subquery(state)
    assert method == "heuristic"


def test_extractor_rejects_echo_of_input(monkeypatch):
    """LLM that just repeats the input wasn't actually helping — fall back."""
    user_input = "Book a cardiologist and what is afib"
    monkeypatch.setattr(
        "nodes.medical_search_node.chat",
        lambda **kwargs: user_input,
    )
    state = {"user_input": user_input, "intents": ["booking", "medical_search"]}
    refined, method = _extract_search_subquery(state)
    assert method == "heuristic"
