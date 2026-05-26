"""Tests for the safety classifier — emergency red-flag detection.

Focus is on the pure `detect_red_flags` function (regex layer). The graph-
integration behavior is exercised by the eval suite. We test both correct
escalation (true positives) and the most-common false positives
(informational questions, historical references) that the informational
guards should suppress.
"""
from __future__ import annotations

import pytest

from nodes.safety import (
    _emergency_response,
    detect_red_flags,
    safety_node,
)

# ---------- true positives ----------

@pytest.mark.parametrize("text, expected_cat", [
    ("I have crushing chest pain right now", "cardiac"),
    ("Help, I think I'm having a heart attack", "cardiac"),
    ("I can't breathe", "cardiac"),
    ("My face is drooping and my speech is slurred", "stroke"),
    ("Sudden numbness on one side of my body", "stroke"),
    ("I want to kill myself", "suicide_self_harm"),
    ("I'm thinking about suicide", "suicide_self_harm"),
    ("I took an overdose", "suicide_self_harm"),
    ("My throat is closing — anaphylaxis", "anaphylaxis"),
    ("I can't stop the bleeding", "severe_bleeding_trauma"),
    ("She is unconscious and not breathing", "altered_mental_status"),
])
def test_detects_red_flag(text, expected_cat):
    cats = detect_red_flags(text)
    assert expected_cat in cats, f"expected {expected_cat} in {cats}"


# ---------- true negatives (informational) ----------

@pytest.mark.parametrize("text", [
    "What are the symptoms of a heart attack?",
    "How do you spot a stroke?",
    "What is anaphylaxis?",
    "My dad had a stroke in 2018, what's his recovery look like?",
    "Family history of heart attack — should I get screened?",
    "I read an article about chest pain warning signs",
    "Book me a cardiologist",
    "Show me Ramesh's history",
    "Update David's record: HbA1c improved",
    "Hello, what can you help me with?",
])
def test_no_false_positive_on_informational_query(text):
    assert detect_red_flags(text) == [], (
        f"unexpected red-flag on informational query: {text!r}"
    )


# ---------- combined emergencies ----------

def test_multiple_categories_combined():
    cats = detect_red_flags("crushing chest pain AND face drooping")
    assert "cardiac" in cats and "stroke" in cats


def test_empty_input_returns_no_flags():
    assert detect_red_flags("") == []
    assert detect_red_flags("   ") == []


# ---------- response template ----------

def test_emergency_response_includes_phone_numbers():
    text = _emergency_response(["cardiac"])
    assert "911" in text
    assert "112" in text
    assert "108" in text


def test_suicide_template_uses_supportive_tone():
    text = _emergency_response(["suicide_self_harm"])
    # Soft opener, not a siren
    assert "💛" in text
    assert "988" in text


def test_suicide_leads_when_combined():
    """Suicide/self-harm should be the first section even if other flags fire."""
    text = _emergency_response(["cardiac", "suicide_self_harm"])
    sui_idx = text.find("988")
    cardiac_idx = text.find("Chew an adult aspirin")
    assert sui_idx >= 0 and cardiac_idx >= 0
    assert sui_idx < cardiac_idx


# ---------- node integration ----------

def test_safety_node_passes_through_on_non_emergency():
    out = safety_node({"user_input": "book me a cardiologist"})
    assert out["is_emergency"] is False
    assert "response" not in out  # composer still does its job


def test_safety_node_short_circuits_on_emergency(monkeypatch):
    # Force the LLM second-opinion to be skipped (return None) so we test
    # the deterministic regex path even if a real LLM key is in the env.
    monkeypatch.setattr("nodes.safety._llm_second_opinion", lambda _t: None)

    out = safety_node({"user_input": "I have crushing chest pain right now"})
    assert out["is_emergency"] is True
    assert "cardiac" in out["emergency_categories"]
    assert "911" in out["response"]
    assert out["intent"] == "emergency"
    assert out["intents"] == ["emergency"]


def test_llm_can_downgrade_borderline_match(monkeypatch):
    monkeypatch.setattr("nodes.safety._llm_second_opinion", lambda _t: False)
    out = safety_node({"user_input": "I had a heart attack 10 years ago"})
    # The regex would otherwise fire on "heart attack" — the historical
    # phrase isn't covered by the informational guards but the LLM corrects
    # the call.
    assert out["is_emergency"] is False
