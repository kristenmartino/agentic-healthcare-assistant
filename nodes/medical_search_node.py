"""Medical search node — searches Medline/WHO/CDC and synthesizes a response."""
from __future__ import annotations

import logging
import re

from config import load_settings
from llm import LLMUnavailable, chat
from prompts import MEDICAL_SEARCH_PROMPT, SEARCH_QUERY_EXTRACTOR_PROMPT
from state import HealthcareState
from tools.medical_search import effective_backend, medical_search

logger = logging.getLogger(__name__)


# Booking / scheduling phrasing to strip when the heuristic falls back.
# These clauses are what break trusted-domain web search — MedlinePlus has
# no page for "book a nephrologist for him". Pattern is conservative: only
# strip clear "Book/Schedule [me/my] [a] <noun> [for X]" clauses.
_BOOKING_CLAUSE = re.compile(
    r"\b(?:book|schedule|set up|arrange)(?:\s+(?:me|my|us|him|her|them))?\s+"
    r"(?:a|an|the)?\s*\w+(?:\s+(?:for|with)\s+(?:me|him|her|them|my\s+\w+))?\s*",
    re.IGNORECASE,
)
_LEADING_CONJUNCTIONS = re.compile(r"^\s*(?:and|then|also|plus)\s+", re.IGNORECASE)


def _heuristic_search_query(text: str) -> str:
    """Strip obvious booking/scheduling phrases. Returns the input unchanged
    when nothing matches (which is the common single-intent case)."""
    cleaned = _BOOKING_CLAUSE.sub(" ", text).strip()
    sentences = [s.strip() for s in re.split(r"[.!?]+", cleaned) if s.strip()]
    sentences = [_LEADING_CONJUNCTIONS.sub("", s) for s in sentences]
    return ". ".join(sentences) or text


def _extract_search_subquery(state: HealthcareState) -> tuple[str, str]:
    """Produce a focused search-friendly query from a multi-intent message.

    The classifier may route a single user input to multiple branches
    (e.g. booking + medical_search). The booking-phrased portion of the
    sentence ("Book me a nephrologist for him") returns zero hits when
    sent to a site:medlineplus.gov-restricted web search. Strip it.

    Returns: (refined_query, method) where method is "llm", "heuristic",
    or "passthrough" (single-intent, no refinement needed).
    """
    user_input = state.get("user_input") or ""
    intents = state.get("intents") or [state.get("intent", "")]

    other_intents = {i for i in intents if i and i != "medical_search"}
    if not other_intents:
        return user_input, "passthrough"

    try:
        refined = chat(
            messages=[
                {"role": "system", "content": SEARCH_QUERY_EXTRACTOR_PROMPT},
                {"role": "user", "content": user_input},
            ],
            temperature=0.0,
            max_tokens=40,
        ).strip().strip('"').strip("'")
        # Sanity floor: between 3 and 120 chars, and shorter than original.
        if (3 <= len(refined) <= 120
                and refined.lower() != user_input.lower()):
            return refined, "llm"
    except LLMUnavailable:
        pass
    except Exception as exc:
        logger.debug("Subquery LLM extraction failed: %s — falling back", exc)

    return _heuristic_search_query(user_input), "heuristic"


def medical_search_node(state: HealthcareState) -> dict:
    settings = load_settings()
    original_query = state.get("user_input") or ""
    query, extraction_method = _extract_search_subquery(state)

    if not query.strip():
        return {
            "medical_info": [],
            "tool_log": [{"node": "medical_search", "result": "skipped",
                          "reason": "empty query"}],
        }

    # 1. Run the search using the (possibly refined) query.
    try:
        results = medical_search(query, top_k=4, tavily_api_key=settings.tavily_api_key)
    except Exception as exc:
        logger.exception("Medical search failed")
        return {
            "error": f"Medical search failed: {exc}",
            "medical_info": [],
            "tool_log": [{"node": "medical_search", "result": "failed", "error": str(exc)}],
        }

    if not results:
        return {
            "medical_info": [],
            "tool_log": [{"node": "medical_search", "result": "empty",
                          "query": query, "original_query": original_query,
                          "extraction_method": extraction_method}],
        }

    # 2. Build sources for the composer + state.sources reducer
    sources = [
        {
            "index": i + 1,
            "title": r["title"],
            "url": r["url"],
            "source": r.get("source", "unknown"),
        }
        for i, r in enumerate(results)
    ]

    # 3. Ask the LLM to synthesize. Pass the ORIGINAL user question so the
    # synthesis remains anchored to what the user actually asked, even though
    # the underlying web search used a stripped-down version.
    snippets_block = "\n\n".join(
        f"[{i+1}] {r['title']}\n{r['snippet']}\nURL: {r['url']}"
        for i, r in enumerate(results)
    )
    user_block = f"User question: {original_query}\n\nSearch results:\n{snippets_block}"

    try:
        synthesis = chat(
            messages=[
                {"role": "system", "content": MEDICAL_SEARCH_PROMPT},
                {"role": "user", "content": user_block},
            ],
            temperature=0.0,
            max_tokens=400,
        ).strip()
    except LLMUnavailable as exc:
        logger.warning("Medical search synthesizer unavailable: %s", exc)
        synthesis = (
            "(No LLM synthesis available — showing raw snippets.)\n\n" + snippets_block
        )

    medical_info = [{"synthesis": synthesis}] + results
    backend = effective_backend(results)
    return {
        "medical_info": medical_info,
        "sources": sources,
        "tool_log": [{
            "node": "medical_search",
            "tool": "medical_search",
            "query": query,
            "original_query": original_query,
            "extraction_method": extraction_method,
            "results_count": len(results),
            "backend": backend,
            "degraded": backend == "stub",
        }],
    }
