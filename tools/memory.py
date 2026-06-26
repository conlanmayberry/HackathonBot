"""Cross-build memory — the one piece of "agent learning" worth borrowing natively.

Every finished build appends a tiny outcome record; before architecting a new spec the
planner recalls the most similar past builds and folds their lessons into the architect
prompt, so specs pass QA on the first try more often (fewer fix rounds = lower cost, less
latency, higher reliability).

Design rules (match the rest of the codebase):
- Pure stdlib, zero new dependencies.
- Best-effort: every function swallows its own errors and degrades to a no-op (empty
  recall / silent record). A memory problem can NEVER break a build.
- Storage is one-JSON-object-per-line under the already-gitignored output/.meta/ dir, so
  it never lands in a generated project or a GitHub push.
- v1 uses simple theme + tech-stack matching (no embeddings). Good enough to be useful;
  swap in semantic recall later if needed.
"""

import json
from datetime import datetime, timezone

from tools.file_writer import META_DIR

HISTORY_FILE = META_DIR / "build_history.jsonl"


def _norm_stack(stack) -> list[str]:
    """Lower-cased, de-whitespaced tech-stack tokens; [] for anything non-list."""
    if not isinstance(stack, (list, tuple)):
        return []
    return [str(s).strip().lower() for s in stack if str(s).strip()]


def record_build(*, theme: str, hackathon: str, idea: dict, spec: dict,
                 qa_result: dict, fix_rounds: int) -> None:
    """Append one compact outcome record for a finished build. Silent on any error."""
    try:
        idea = idea or {}
        qa_result = qa_result or {}
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "theme": (theme or "").strip(),
            "hackathon": (hackathon or "").strip(),
            "idea_title": (idea.get("title") or "").strip(),
            "tech_stack": _norm_stack(idea.get("tech_stack")),
            "spec_slug": (spec.get("slug") or "") if isinstance(spec, dict) else "",
            "qa_passed": bool(qa_result.get("passed")),
            "fix_rounds": int(fix_rounds),
            "backend_failed": bool(qa_result.get("backend_failures")),
            "frontend_failed": bool(qa_result.get("frontend_failures")),
        }
        META_DIR.mkdir(parents=True, exist_ok=True)
        with HISTORY_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:  # noqa: BLE001 — memory is best-effort; never break a build
        pass


def _load_history(limit: int = 500) -> list[dict]:
    """Most recent ``limit`` records (oldest→newest). [] if absent/unreadable."""
    try:
        if not HISTORY_FILE.exists():
            return []
        out: list[dict] = []
        for line in HISTORY_FILE.read_text(encoding="utf-8").splitlines()[-limit:]:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out
    except OSError:
        return []


def recall_relevant(theme: str, tech_stack, limit: int = 5) -> list[dict]:
    """Up to ``limit`` past builds most relevant to this theme + stack, best first.
    Relevance = theme match (exact/substring) + tech-stack Jaccard overlap; ties broken
    by recency. Records with no theme/stack relevance are excluded entirely."""
    history = _load_history()
    if not history:
        return []
    theme_l = (theme or "").strip().lower()
    want = set(_norm_stack(tech_stack))

    def relevance(rec: dict) -> float:
        score = 0.0
        rt = (rec.get("theme") or "").lower()
        if theme_l and rt:
            if theme_l == rt:
                score += 3.0
            elif theme_l in rt or rt in theme_l:
                score += 1.5
        have = set(rec.get("tech_stack") or [])
        if want and have:
            score += 2.0 * len(want & have) / len(want | have)  # Jaccard
        return score

    scored = []
    for idx, rec in enumerate(history):  # idx encodes recency (higher = newer)
        rel = relevance(rec)
        if rel > 0:
            scored.append((rel, idx, rec))
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [rec for _, _, rec in scored[:limit]]


def summarize_lessons(records: list[dict]) -> str:
    """Render a short advisory block for the architect prompt. Empty string when there's
    no relevant history (the caller then injects nothing)."""
    if not records:
        return ""
    n = len(records)
    passed = sum(1 for r in records if r.get("qa_passed"))
    avg_rounds = sum(int(r.get("fix_rounds") or 0) for r in records) / n
    backend_iss = sum(1 for r in records if r.get("backend_failed"))
    frontend_iss = sum(1 for r in records if r.get("frontend_failed"))

    lines = [
        f"You have {n} past build(s) on similar themes/stacks to learn from:",
        f"- {passed}/{n} passed QA; {avg_rounds:.1f} fix round(s) needed on average.",
    ]
    if backend_iss:
        lines.append(
            f"- {backend_iss} hit BACKEND failures in QA — be especially precise about the "
            "api_endpoints contract (exact paths/methods and request/response shapes) and the "
            "Pydantic models, so the frontend's calls match the server on the first try."
        )
    if frontend_iss:
        lines.append(
            f"- {frontend_iss} hit FRONTEND failures in QA — make sure the file_manifest lists "
            "EVERY js module and asset the HTML references, so nothing is referenced-but-missing."
        )
    clean = [r for r in records
             if r.get("qa_passed") and int(r.get("fix_rounds") or 0) == 0]
    if clean:
        ex = ", ".join(
            f"\"{r.get('idea_title')}\" ({'/'.join(r.get('tech_stack') or []) or 'n/a'})"
            for r in clean[:2]
        )
        lines.append(f"- Past builds that passed QA cleanly (good patterns to lean on): {ex}.")
    return "\n".join(lines)
