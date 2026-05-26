"""Safety classifier — emergency red-flag detection.

Inserted between START and `classify_intent`. If the user's input matches any
of a curated set of emergency phrases, the graph short-circuits straight to
the composer with a hardcoded urgent-care template. The LLM is never asked,
because the worst failure mode for a clinical assistant is "calm advice on a
medical emergency" — that's negligence, not a hallucination.

Two layers:

1. **Deterministic regex sweep** (this file). Always runs, always under 1ms.
   Sourced from the AHA STEMI symptom list, the National Suicide Prevention
   Lifeline crisis criteria, and the WHO sudden-onset stroke checklist. False
   positives are *fine* — directing someone to 911 who didn't strictly need
   it is a tolerable harm; missing an MI is not.

2. **LLM second opinion** (optional). When an LLM is configured, run a quick
   second-opinion classification on borderline cases (regex says "maybe"; LLM
   decides yes/no). Disabled in stub mode because the stub LLM can't reason
   about clinical urgency.

Output:
- `is_emergency`: bool, set on the state for downstream nodes.
- `emergency_categories`: list of which red-flag groups matched.
- When emergency: a hardcoded `response`, no further routing.

Caller policy in graph.py: when `is_emergency=True`, route from this node
directly to END (or to compose_response with no other branches).
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from state import HealthcareState

logger = logging.getLogger(__name__)


# ---------- red-flag dictionaries ----------
#
# Each tuple is (category, list_of_regexes). Regexes are word-boundary anchored
# where it matters; case-insensitive at match time.

_RED_FLAGS: dict[str, list[str]] = {
    "cardiac": [
        r"\bchest pain\b",
        r"\bcrushing (?:chest|pressure)\b",
        r"\bpressure (?:in|on) (?:my )?chest\b",
        r"\bheart attack\b",
        r"\bcan(?:'t| not) breathe\b",
        r"\bshortness of breath\b.{0,30}\b(?:sweat|nausea|left arm|jaw)\b",
        r"\b(?:left|both) arm.{0,20}\bnumb\b",
    ],
    "stroke": [
        r"\bstroke\b",
        r"\bface.{0,15}drooping\b",
        r"\bdrooping.{0,15}face\b",
        r"\bslurred speech\b",
        r"\b(?:can(?:'t| not)|trouble) (?:speak|talk)\b",
        r"\bsudden(?:ly)?\b.{0,30}\b(?:numbness|weakness|confusion|dizz)\b",
        r"\bone side.{0,20}\b(?:numb|weak|paralys)\b",
    ],
    "suicide_self_harm": [
        r"\bsuicid",
        r"\bkill myself\b",
        r"\bend my life\b",
        r"\b(?:don'?t|do not) want to (?:live|be alive)\b",
        r"\bharm(?:ing)? myself\b",
        r"\boverdos",
    ],
    "anaphylaxis": [
        r"\banaphyla",
        r"\bthroat (?:closing|swelling)\b",
        r"\b(?:tongue|lips) (?:swelling|swollen)\b",
        r"\bsevere allergic reaction\b",
    ],
    "severe_bleeding_trauma": [
        r"\b(?:can(?:'t| not)) stop (?:the )?bleeding\b",
        r"\bsevere bleeding\b",
        r"\bhemorrh",
        r"\bbleeding (?:heavily|profusely)\b",
        r"\bcompound fracture\b",
    ],
    "altered_mental_status": [
        r"\bunconscious\b",
        r"\bnot breathing\b",
        r"\bno pulse\b",
        r"\bseizure\b.{0,30}\b(?:now|right now|happening|ongoing)\b",
    ],
}


# Phrases that look red-flag-y but are clearly informational/historical.
# Stripping these *before* the red-flag sweep avoids the most common false
# positives: questions about conditions vs. lived experience of them.
_INFORMATIONAL_GUARDS: list[str] = [
    r"\bwhat (?:is|are) (?:a |the )?",                  # "what is a heart attack"
    r"\bhow (?:do|to) (?:i |you )?(?:recognize|spot|identify)\b",
    r"\b(?:symptoms? of|signs? of|causes of)\b",        # "symptoms of stroke"
    r"\b(?:history of|previous|past)\b.{0,30}\b(?:stroke|heart attack|seizure|suicid)\b",
    r"\b(?:family history of|risk for)\b",
    r"\b(?:read|article|book) about\b",
    r"\bhad a (?:stroke|heart attack|seizure)\b.{0,30}\b(?:in|years? ago|last|previously|history)\b",
]


def _matches(text: str, patterns: list[str]) -> list[str]:
    """Return the list of patterns that fired."""
    return [p for p in patterns if re.search(p, text, re.IGNORECASE)]


def _is_informational(text: str) -> bool:
    """Cheap heuristic: is the user asking *about* a condition rather than
    reporting their current experience?"""
    return any(re.search(p, text, re.IGNORECASE) for p in _INFORMATIONAL_GUARDS)


def detect_red_flags(text: str) -> list[str]:
    """Return the list of category names that matched. Empty if none.

    Pure function — no side effects, no LLM. Safe to unit-test.
    """
    if not text:
        return []
    if _is_informational(text):
        return []
    hits: list[str] = []
    for category, patterns in _RED_FLAGS.items():
        if _matches(text, patterns):
            hits.append(category)
    return hits


# ---------- emergency response template ----------

_EMERGENCY_RESPONSES: dict[str, str] = {
    "cardiac": (
        "🚨 **This sounds like a possible cardiac emergency.**\n\n"
        "**If you or the person experiencing this are not already in a hospital, "
        "call emergency services immediately:**\n"
        "- **US**: 911\n"
        "- **UK / EU**: 112 or 999\n"
        "- **India**: 108\n\n"
        "Chew an adult aspirin (325 mg) while you wait if there are no allergies "
        "or bleeding concerns, and stay seated or lying down. Do not drive yourself."
    ),
    "stroke": (
        "🚨 **This sounds like a possible stroke.**\n\n"
        "Time matters — every minute counts. **Call emergency services now:**\n"
        "- **US**: 911\n"
        "- **UK / EU**: 112 or 999\n"
        "- **India**: 108\n\n"
        "Note the time the symptoms started — clinicians will need this to choose "
        "treatment. Do not eat, drink, or take medication while waiting."
    ),
    "suicide_self_harm": (
        "💛 **I hear you, and I'm glad you reached out.**\n\n"
        "Please contact a crisis support line right now — they're free, "
        "confidential, and trained for exactly this:\n"
        "- **US**: call or text **988** (Suicide & Crisis Lifeline)\n"
        "- **UK**: **116 123** (Samaritans)\n"
        "- **India**: **iCall 9152987821**, or **Vandrevala Foundation 1860-2662-345**\n"
        "- **International directory**: https://findahelpline.com\n\n"
        "If you have means to harm yourself nearby, please move away from them "
        "or have someone secure them while you talk to the line."
    ),
    "anaphylaxis": (
        "🚨 **This sounds like anaphylaxis — a life-threatening allergic reaction.**\n\n"
        "- If an **EpiPen** or auto-injector is available, use it now (outer thigh).\n"
        "- **Call emergency services**: 911 (US) / 112 or 999 (UK/EU) / 108 (India).\n"
        "- Even after EpiPen use, go to the ER — symptoms can return."
    ),
    "severe_bleeding_trauma": (
        "🚨 **This sounds like a serious bleeding emergency.**\n\n"
        "- **Call emergency services**: 911 (US) / 112 or 999 (UK/EU) / 108 (India).\n"
        "- Apply firm direct pressure with a clean cloth.\n"
        "- Do not remove embedded objects; press around them.\n"
        "- Keep the injured area elevated if possible."
    ),
    "altered_mental_status": (
        "🚨 **This sounds like a medical emergency requiring immediate care.**\n\n"
        "- **Call emergency services**: 911 (US) / 112 or 999 (UK/EU) / 108 (India).\n"
        "- If unconscious, check breathing; if not breathing, start CPR if trained.\n"
        "- Place an unresponsive person on their side (recovery position) if they "
        "are breathing."
    ),
}


_DEFAULT_EMERGENCY_FOOTER = (
    "\n\n---\n*This assistant cannot replace emergency clinical care. "
    "Please contact emergency services right now if you have any doubt.*"
)


def _emergency_response(categories: list[str]) -> str:
    """Compose the hardcoded response by stitching category templates."""
    # Suicide/self-harm always leads — softer tone, different first instinct
    ordered = sorted(categories, key=lambda c: 0 if c == "suicide_self_harm" else 1)
    sections = [_EMERGENCY_RESPONSES[c] for c in ordered if c in _EMERGENCY_RESPONSES]
    return "\n\n".join(sections) + _DEFAULT_EMERGENCY_FOOTER


# ---------- LLM second-opinion (optional) ----------

_LLM_SAFETY_PROMPT = (
    "You are a clinical-safety triage filter. Given a user message, decide if "
    "it describes a *current* medical emergency that needs immediate 911-level "
    "care. Examples that DO need 911: 'crushing chest pain right now', 'I am "
    "having a stroke', 'my child isn't breathing'. Examples that DO NOT need "
    "911: 'what are the symptoms of a heart attack?', 'my dad had a stroke in "
    "2018', 'book me a cardiologist'. Reply with exactly one word: EMERGENCY "
    "or NORMAL."
)


def _llm_second_opinion(user_input: str) -> Optional[bool]:
    """Ask the configured LLM to confirm an emergency. Returns None on failure
    or stub provider (caller falls back to regex result)."""
    try:
        from config import load_settings
        from llm import LLMUnavailable, chat
        if load_settings().llm_provider == "stub":
            return None
        raw = chat(
            messages=[
                {"role": "system", "content": _LLM_SAFETY_PROMPT},
                {"role": "user", "content": user_input},
            ],
            temperature=0.0,
            max_tokens=4,
        ).strip().upper()
        if raw.startswith("EMERGENCY"):
            return True
        if raw.startswith("NORMAL"):
            return False
        return None
    except Exception as exc:  # LLMUnavailable + transport errors
        logger.debug("Safety LLM second-opinion unavailable: %s", exc)
        return None


# ---------- the node ----------

def safety_node(state: HealthcareState) -> dict:
    """Pre-classifier safety check. Returns short-circuit response if emergency."""
    user_input = (state.get("user_input") or "").strip()

    categories = detect_red_flags(user_input)

    # If regex fired, optionally confirm with the LLM. We *only* downgrade
    # ("NORMAL") if the LLM is confident; on unknown we keep the regex hit.
    if categories:
        llm_result = _llm_second_opinion(user_input)
        if llm_result is False:
            logger.info("Safety: regex matched but LLM downgraded to NORMAL")
            categories = []

    if not categories:
        return {
            "is_emergency": False,
            "tool_log": [{
                "node": "safety",
                "is_emergency": False,
                "categories": [],
            }],
        }

    response = _emergency_response(categories)
    return {
        "is_emergency": True,
        "emergency_categories": categories,
        "response": response,
        "intent": "emergency",
        "intents": ["emergency"],
        "tool_log": [{
            "node": "safety",
            "is_emergency": True,
            "categories": categories,
        }],
    }
