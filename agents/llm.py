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

# Two tiers, picked per task to balance quality against cost:
#   MODEL      — the architect's brain. Opus 4.8 for the high-stakes reasoning where a
#                wrong call cascades through the whole build: Devpost research + idea
#                generation, and the build-spec/architecture step the entire team builds against.
#   MODEL_CODE — the builders' hands. Sonnet 4.6 is a top-tier coding model at ~40% lower
#                cost ($3/$15 vs $5/$25 per 1M in/out); used for the well-scoped work that
#                builds against a clear spec: frontend/backend dev, QA tests, README glue,
#                and the interactive chat editor. It accepts the identical thinking API.
MODEL = "claude-opus-4-8"
MODEL_CODE = "claude-sonnet-4-6"

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
    cache: bool = False,
):
    """Stream a completion, yielding ``(kind, delta)`` tuples.

    ``kind`` is ``"thinking"`` for adaptive-thinking deltas and ``"text"`` for the
    actual answer. Callers accumulate only the ``"text"`` deltas to get the final
    response; both kinds are usually surfaced to the UI as live thoughts.

    ``thinking=True`` enables adaptive thinking; ``effort`` (low|medium|high|xhigh|max)
    sets reasoning depth. ``temperature`` is intentionally unsupported (400 on this model).

    ``cache=True`` marks the ``system`` prompt as a prompt-caching breakpoint
    (``cache_control: ephemeral``). Put the large STABLE context (e.g. the shared build
    spec) in ``system`` and the volatile, per-call task in ``messages``: the cached prefix
    is then reused across calls within the cache TTL (e.g. a dev's run→repair), cutting
    input cost and time-to-first-token. Caching only triggers above the model's minimum
    cacheable prefix; below it the flag is a harmless no-op. Set ``LLM_DEBUG=1`` to log
    cache hit/miss token counts to stderr.
    """
    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system:
        # When caching, send ``system`` as a content block carrying a cache breakpoint so
        # everything up to and including it is cached; otherwise the plain string is fine.
        kwargs["system"] = (
            [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
            if cache else system
        )
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
        if os.getenv("LLM_DEBUG"):
            _log_cache_usage(model, await s.get_final_message())


def _log_cache_usage(model: str, message) -> None:
    """Print prompt-cache token accounting for one call (only when LLM_DEBUG is set).
    ``cache_read_input_tokens`` > 0 means a cache hit; ``cache_creation_input_tokens``
    is the prefix we just wrote (billed ~25% over base input, read back at ~10%)."""
    import sys
    u = getattr(message, "usage", None)
    if u is None:
        return
    print(
        f"[llm.cache] {model}: input={getattr(u, 'input_tokens', 0)} "
        f"cache_write={getattr(u, 'cache_creation_input_tokens', 0)} "
        f"cache_read={getattr(u, 'cache_read_input_tokens', 0)} "
        f"output={getattr(u, 'output_tokens', 0)}",
        file=sys.stderr,
    )
