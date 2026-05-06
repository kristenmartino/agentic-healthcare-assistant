"""Medical search node — searches Medline/WHO/CDC and synthesizes a response."""
from __future__ import annotations

import logging

from config import load_settings
from llm import LLMUnavailable, chat
from prompts import MEDICAL_SEARCH_PROMPT
from state import HealthcareState
from tools.medical_search import medical_search

logger = logging.getLogger(__name__)


def medical_search_node(state: HealthcareState) -> dict:
    settings = load_settings()
    query = state.get("user_input") or ""

    if not query.strip():
        return {
            "medical_info": [],
            "tool_log": [{"node": "medical_search", "result": "skipped", "reason": "empty query"}],
        }

    # 1. Run the search
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
            "tool_log": [{"node": "medical_search", "result": "empty", "query": query}],
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

    # 3. Ask the LLM to synthesize. The composer also has access to results, but
    # synthesizing here keeps the per-branch concern self-contained.
    snippets_block = "\n\n".join(
        f"[{i+1}] {r['title']}\n{r['snippet']}\nURL: {r['url']}"
        for i, r in enumerate(results)
    )
    user_block = f"User question: {query}\n\nSearch results:\n{snippets_block}"

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

    # We attach the synthesis as a special pseudo-result so the composer has it.
    medical_info = [{"synthesis": synthesis}] + results

    return {
        "medical_info": medical_info,
        "sources": sources,
        "tool_log": [{
            "node": "medical_search",
            "tool": "medical_search",
            "query": query,
            "results_count": len(results),
            "backend": results[0].get("source") if results else "none",
        }],
    }
