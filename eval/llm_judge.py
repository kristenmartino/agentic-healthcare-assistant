"""LLM-as-judge layer for response quality scoring.

The base eval (`eval/qa_eval.py`) grades routing + state shape — deterministic
checks that don't depend on prose quality. This module adds a complementary
layer: it asks the same LLM (or a different one if `JUDGE_MODEL` is set) to
score each response on three dimensions, returning structured JSON.

Dimensions (each scored 1-5):
- relevance: did the response address the user's actual ask?
- accuracy: are all factual claims supported by the branch outputs (no
  hallucinated doctor names, dates, confirmation numbers)?
- tone: is the tone appropriate for a healthcare context (warm, professional,
  with the disclaimer)?

In stub mode this returns 0/0/0 placeholders (the templated outputs aren't
representative of real quality, so judging them is misleading).

Run: `python -m eval.llm_judge` after `python -m eval.qa_eval` has produced
a results file. The most recent results file is auto-loaded.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# Allow running as script from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_settings
from graph import build_workflow
from llm import LLMUnavailable, chat
from eval.qa_eval import GROUND_TRUTH

logger = logging.getLogger(__name__)


JUDGE_PROMPT = """You are an evaluator for a healthcare AI assistant.

You will be given:
1. A user's query.
2. The expected intent(s) the system should have routed to.
3. The structured branch outputs the system actually produced (appointment, record, history, search results).
4. The final composed response shown to the user.

Score the response on three dimensions, each 1 to 5:
- relevance: did the response address the user's actual ask? (5 = fully on-topic, 1 = off-topic)
- accuracy: are all factual claims supported by the branch outputs? Check that doctor names, dates, confirmation numbers, patient details match the branch outputs. (5 = nothing invented, 1 = significant fabrication)
- tone: is the tone appropriate for a healthcare context (warm, professional, includes a clinical-care disclaimer)? (5 = excellent tone, 1 = inappropriate tone)

Output ONLY a JSON object with this exact shape, no other text:
{"relevance": <1-5>, "accuracy": <1-5>, "tone": <1-5>, "rationale": "<one sentence>"}
"""


def _extract_json(text: str) -> dict | None:
    """Pull the first {...} JSON block out of an LLM response."""
    text = text.strip()
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Find first balanced object
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def judge_one(
    *,
    query: str,
    expected_intents: list[str],
    state: dict,
    response: str,
) -> dict:
    """Ask the LLM to score one (query, response) pair."""
    # Build a compact rendering of branch outputs
    branch_outputs: dict = {}
    for k in ("appointment", "record_change", "history_summary", "medical_info", "intents", "patient_name"):
        v = state.get(k)
        if v:
            branch_outputs[k] = v

    user_block = (
        f"=== Query ===\n{query}\n\n"
        f"=== Expected intents ===\n{expected_intents}\n\n"
        f"=== Branch outputs ===\n{json.dumps(branch_outputs, default=str, indent=2)[:2000]}\n\n"
        f"=== Final response ===\n{response}\n\n"
        f"Score now."
    )

    try:
        raw = chat(
            messages=[
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": user_block},
            ],
            temperature=0.0,
            max_tokens=200,
        )
    except LLMUnavailable as exc:
        return {
            "relevance": 0,
            "accuracy": 0,
            "tone": 0,
            "rationale": f"LLM unavailable: {exc}",
            "raw": "",
        }

    parsed = _extract_json(raw)
    if not parsed:
        return {
            "relevance": 0,
            "accuracy": 0,
            "tone": 0,
            "rationale": f"Could not parse judge output: {raw[:200]}",
            "raw": raw,
        }

    # Coerce to int + clamp
    out = {}
    for k in ("relevance", "accuracy", "tone"):
        try:
            out[k] = max(1, min(5, int(parsed.get(k, 0))))
        except (TypeError, ValueError):
            out[k] = 0
    out["rationale"] = str(parsed.get("rationale", ""))[:300]
    out["raw"] = raw[:500]
    return out


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    settings = load_settings()

    if settings.llm_provider == "stub":
        print("⚠️  LLM provider is 'stub'. Skipping LLM-judge — set OPENAI_API_KEY or GROQ_API_KEY first.")
        return 1

    # Reset EHR for determinism
    if settings.ehr_backend == "sqlite":
        from tools.ehr_db import initialize_ehr
        if Path(settings.records_xlsx_path).exists():
            initialize_ehr(settings.records_xlsx_path, settings.ehr_db_path)
    elif settings.ehr_backend == "fhir_fixture":
        writes = Path(settings.fhir_fixture_dir) / "patients_writes.json"
        if writes.exists():
            writes.unlink()
        from tools.ehr import clear_backend_cache
        clear_backend_cache()

    print("=" * 70)
    print(f" LLM-Judge Evaluation — Healthcare Assistant")
    print(f" Provider: {settings.llm_provider} ({settings.llm_model})")
    print(f" Judge model: same (set JUDGE_MODEL to override — not yet wired)")
    print(f" {len(GROUND_TRUTH)} cases × 3 dimensions = {len(GROUND_TRUTH) * 3} scores")
    print("=" * 70)

    workflow = build_workflow(with_checkpoint=False)

    judged: list[dict] = []
    for i, gt in enumerate(GROUND_TRUTH, 1):
        thread_id = f"judge-{gt['id']}"
        # Run the workflow
        try:
            result = workflow.invoke(
                {"user_input": gt["query"], "history": []},
                config={"configurable": {"thread_id": thread_id}},
            )
        except Exception as exc:
            print(f"[{i:2d}/{len(GROUND_TRUTH)}] {gt['id']:12s} workflow error: {exc}")
            judged.append({"id": gt["id"], "error": str(exc)})
            continue

        # Judge it
        scores = judge_one(
            query=gt["query"],
            expected_intents=gt["expected_intents"],
            state=dict(result),
            response=result.get("response", ""),
        )

        avg = (scores["relevance"] + scores["accuracy"] + scores["tone"]) / 3
        print(
            f"[{i:2d}/{len(GROUND_TRUTH)}] {gt['id']:12s}  "
            f"R={scores['relevance']} A={scores['accuracy']} T={scores['tone']} "
            f"(avg {avg:.1f}/5) — {scores['rationale'][:70]}"
        )
        judged.append({
            "id": gt["id"],
            "query": gt["query"],
            "scores": scores,
            "response_excerpt": (result.get("response") or "")[:300],
        })

    # Aggregate
    valid = [j for j in judged if "scores" in j]
    if not valid:
        print("No valid judgments produced.")
        return 1

    aggregated = {
        "relevance_avg": sum(j["scores"]["relevance"] for j in valid) / len(valid),
        "accuracy_avg": sum(j["scores"]["accuracy"] for j in valid) / len(valid),
        "tone_avg": sum(j["scores"]["tone"] for j in valid) / len(valid),
    }
    aggregated["overall_avg"] = sum(aggregated.values()) / 3

    print("=" * 70)
    summary = {
        "timestamp": datetime.now().isoformat(),
        "provider": settings.llm_provider,
        "model": settings.llm_model,
        "cases_judged": len(valid),
        **{k: round(v, 2) for k, v in aggregated.items()},
    }
    print(json.dumps(summary, indent=2))
    print("=" * 70)

    out_path = Path(__file__).parent / f"llm_judge_{int(time.time())}.json"
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "cases": judged}, f, indent=2, default=str)
    print(f"Results: {out_path}")

    return 0 if aggregated["overall_avg"] >= 3.5 else 1


if __name__ == "__main__":
    sys.exit(main())
