"""Medical-info search wrapper.

Tries Tavily first (if TAVILY_API_KEY is set; better quality, real medical sources),
falls back to DuckDuckGo (no key required, rate-limited).

Both backends constrain queries to trusted domains:
medlineplus.gov, who.int, cdc.gov, nih.gov, mayoclinic.org.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


_TRUSTED_DOMAINS = (
    "medlineplus.gov",
    "who.int",
    "cdc.gov",
    "nih.gov",
    "mayoclinic.org",
    "nhs.uk",
)

# Site filter for DuckDuckGo / Tavily query construction
_SITE_FILTER = " OR ".join(f"site:{d}" for d in _TRUSTED_DOMAINS)


def medical_search(
    query: str,
    *,
    top_k: int = 4,
    tavily_api_key: str | None = None,
) -> list[dict]:
    """Run a medical-info search. Returns list of {title, snippet, url, source}.

    Tries Tavily first if a key is provided, then DuckDuckGo, then a deterministic
    stub (so the graph still completes when no internet is available).
    """
    tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")

    if tavily_api_key:
        try:
            return _tavily_search(query, tavily_api_key, top_k)
        except Exception as exc:
            logger.warning("Tavily search failed: %s — falling back to DuckDuckGo", exc)

    try:
        return _ddg_search(query, top_k)
    except Exception as exc:
        logger.warning("DuckDuckGo search failed: %s — falling back to stub", exc)

    return _stub_search(query, top_k)


def _tavily_search(query: str, api_key: str, top_k: int) -> list[dict]:
    """Use Tavily API directly via requests (no SDK dep)."""
    import requests

    response = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": api_key,
            "query": f"{query} ({_SITE_FILTER})",
            "max_results": top_k,
            "search_depth": "basic",
            "include_domains": list(_TRUSTED_DOMAINS),
        },
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()
    results = []
    for r in data.get("results", [])[:top_k]:
        results.append({
            "title": r.get("title", ""),
            "snippet": r.get("content", "")[:500],
            "url": r.get("url", ""),
            "source": "tavily",
        })
    return results


def _ddg_search(query: str, top_k: int) -> list[dict]:
    """DuckDuckGo via ddgs (new) or duckduckgo_search (old). No API key required.

    Tries the newer `ddgs` package first; falls back to the deprecated
    `duckduckgo_search` if `ddgs` isn't installed or trips an SSL/TLS issue
    (some Python builds + LibreSSL combos can't negotiate TLS 1.3).
    """
    DDGS = None
    last_exc: Exception | None = None

    try:
        from ddgs import DDGS as _DDGS  # type: ignore
        DDGS = _DDGS
    except Exception as exc:  # ImportError, ValueError (TLS), etc.
        last_exc = exc
        try:
            from duckduckgo_search import DDGS as _DDGS  # type: ignore
            DDGS = _DDGS
        except ImportError as exc2:
            raise RuntimeError(
                "Neither `ddgs` nor `duckduckgo_search` is installed. "
                "Install one: `pip install ddgs` (preferred) "
                "or `pip install duckduckgo-search` (legacy)."
            ) from exc2

    full_query = f"{query} {_SITE_FILTER}"
    results: list[dict] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(full_query, max_results=top_k):
                # Field names: ddgs uses 'title'/'body'/'href'; new ddgs may use 'description'/'url'
                title = r.get("title") or r.get("name") or ""
                snippet = r.get("body") or r.get("description") or r.get("snippet") or ""
                url = r.get("href") or r.get("url") or r.get("link") or ""
                results.append({
                    "title": title,
                    "snippet": str(snippet)[:500],
                    "url": url,
                    "source": "duckduckgo",
                })
    except Exception as exc:
        # Re-raise so the caller falls through to stub_search
        if last_exc and not results:
            raise RuntimeError(f"ddgs failed ({last_exc}), legacy DDG also failed: {exc}") from exc
        raise

    return results


def _stub_search(query: str, top_k: int) -> list[dict]:
    """Deterministic placeholder when no real backend is reachable.

    Keeps the graph testable end-to-end without internet. The returned
    snippets are clearly marked as stub so the composer can flag them.
    """
    return [
        {
            "title": f"[stub] Overview of '{query}'",
            "snippet": (
                "[STUB SEARCH RESULT — no internet/Tavily/DuckDuckGo available]. "
                f"In a real run this would be a synthesized snippet from MedlinePlus or WHO "
                f"about: {query}."
            ),
            "url": "https://medlineplus.gov/",
            "source": "stub",
        }
    ][:top_k]


# CLI: python -m tools.medical_search "query"
if __name__ == "__main__":
    import json as _json
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if len(sys.argv) < 2:
        print("Usage: python -m tools.medical_search 'your query'")
        sys.exit(1)
    print(_json.dumps(medical_search(" ".join(sys.argv[1:])), indent=2))
