"""Shared Anthropic streaming helper.

Every agent in HackathonBot streams through this so behaviour is consistent:
- a single shared async client (connection pooling, generous timeout + retries),
- optional *adaptive thinking* so the model genuinely reasons before producing
  output (this is what makes the agents "take their time" instead of dashing),
- a uniform (kind, delta) event stream the agents turn into UI "thoughts".

Usage:

    full_text = ""
    async for kind, delta in stream(model=MODEL, max_tokens=32000,
                                    messages=[...], thinking=True, effort="high"):
        if kind == "text":
            full_text += delta
        yield _thought("backend_dev", delta)

claude-opus-4-8 specifics (do NOT regress these — each is a 400 otherwise):
- Thinking is adaptive-only: ``thinking={"type": "adaptive"}``. The old
  ``{"type": "enabled", "budget_tokens": N}`` form is rejected.
- Reasoning depth is controlled by ``output_config={"effort": ...}``
  (low|medium|high|xhigh|max), NOT a token budget.
- ``temperature`` / ``top_p`` / ``top_k`` are removed — never send them.
- ``display: "summarized"`` is set so thinking streams as readable text (the
  default, "omitted", would stream empty thinking deltas).
"""

import os
import anthropic

# The single most capable Claude model — every agent uses it (per project spec).
MODEL = "claude-opus-4-8"

_client: anthropic.AsyncAnthropic | None = None


def client() -> anthropic.AsyncAnthropic:
    """Lazily build one shared async client. Long timeout because extended
    thinking + large outputs can legitimately take a couple of minutes."""
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            timeout=600.0,
            max_retries=3,
        )
    return _client


async def stream(
    *,
    model: str = MODEL,
    max_tokens: int,
    messages: list[dict],
    system: str | None = None,
    thinking: bool = False,
    effort: str | None = None,
):
    """Stream a completion, yielding ``(kind, delta)`` tuples.

    ``kind`` is ``"thinking"`` for adaptive-thinking deltas and ``"text"`` for the
    actual answer. Callers accumulate only the ``"text"`` deltas to get the final
    response; both kinds are usually surfaced to the UI as live thoughts.

    ``thinking=True`` enables adaptive thinking; ``effort`` (low|medium|high|xhigh|max)
    sets reasoning depth. ``temperature`` is intentionally unsupported (400 on this model).
    """
    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system:
        kwargs["system"] = system
    if thinking:
        kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
    if effort:
        kwargs["output_config"] = {"effort": effort}

    async with client().messages.stream(**kwargs) as s:
        async for event in s:
            if getattr(event, "type", None) != "content_block_delta":
                continue
            delta = event.delta
            dtype = getattr(delta, "type", None)
            if dtype == "thinking_delta":
                yield ("thinking", delta.thinking)
            elif dtype == "text_delta":
                yield ("text", delta.text)
