import os

from agents.llm import stream, MODEL_CODE
from tools.file_writer import get_output_path, parse_file_blocks, write_file, list_output_files

SYSTEM = """You are the HackathonBot assistant — a senior full-stack developer who makes REAL code \
changes to hackathon projects that have already been generated.

The COMPLETE current contents of every project file are given to you below in the context. You \
ALWAYS have the code — so NEVER ask the user to paste files or share contents; just read what's \
provided and make the change.

When the user asks for a change (new feature, UI tweak, bug fix, new endpoint, style change, etc.):
1. Read the relevant files from the context below.
2. Output the FULL updated content for every file you change, using this EXACT format:
   ===FILE: relative/path/to/file.ext===
   <complete file contents>
   ===END===
   Paths are relative to the project root (e.g. "frontend/story.js", "backend/main.py").
3. After the file blocks, write a SHORT plain-English summary of what you changed and why.

Rules:
- Always output the COMPLETE file, not a diff or snippet — the block REPLACES the whole file.
- Only output blocks for files you actually change. To add a new file, just use a new ===FILE:=== block.
- If the user asks a pure question (no code change needed), just answer — no file blocks.
- Never claim you can't see the code; it is always below. Never ask the user to send files.
- Only decline if the request truly means regenerating the entire project from scratch \
(e.g. "rewrite everything in React") — then say so and explain what re-running would do.
- Be concise. Make the change. Don't ask for permission."""


def _build_context(job: dict) -> str:
    req = job.get("request", {})
    result = job.get("result") or {}

    parts = [
        f"Hackathon: {req.get('hackathon', 'N/A')}",
        f"University: {req.get('university', 'N/A')}",
        f"Theme: {req.get('theme', 'N/A')}",
    ]

    selected = result.get("selected_idea")
    if selected:
        parts.append(f"Project: {selected.get('title')} — {selected.get('description', '')}")
        parts.append(f"Tech stack: {', '.join(selected.get('tech_stack', []))}")

    spec = result.get("build_spec", {})
    if isinstance(spec, dict) and spec.get("summary"):
        parts.append(f"Build spec summary: {spec['summary']}")
    endpoints = spec.get("api_endpoints") if isinstance(spec, dict) else None
    if endpoints:
        ep = ", ".join(f"{e.get('method')} {e.get('path')}" for e in endpoints[:12])
        parts.append(f"API endpoints: {ep}")

    run = result.get("run", {})
    if run:
        parts.append(f"Ports: backend {run.get('backend_port')}, frontend {run.get('frontend_port')}.")

    if selected:
        output_path = get_output_path(selected["title"])
        # Normalize to forward slashes so paths match the ===FILE: path=== format the model
        # must emit (and read cleanly on Windows, where list_output_files returns backslashes).
        files = [f.replace("\\", "/") for f in list_output_files(selected["title"])]
        if files:
            parts.append(f"\nProject files ({len(files)} total):\n" + "\n".join(f"  {f}" for f in files))

        # Inline COMPLETE file contents so the assistant can edit precisely and never needs
        # to ask the user to paste code. Opus 4.8 has a 1M-token context window, so the whole
        # (small) hackathon project fits easily. We NEVER truncate a file mid-content — half a
        # file is useless when the rule is "output the full file". The budget is only a safety
        # ceiling; whole files are included until it's hit, then the rest are listed by name.
        BUDGET = 400_000  # chars (~100k tokens); real projects are far smaller
        used = 0
        skipped: list[str] = []

        def _prio(f: str) -> tuple:
            is_code = f.endswith((".py", ".js", ".jsx", ".ts", ".tsx", ".html", ".css"))
            return (0 if is_code else 1, f)

        file_blocks = []
        for rel in sorted(files, key=_prio):
            try:
                content = (output_path / rel).read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue  # missing or binary — skip
            if used + len(content) > BUDGET:
                skipped.append(rel)
                continue
            lang = rel.rsplit(".", 1)[-1] if "." in rel else ""
            file_blocks.append(f"### {rel}\n```{lang}\n{content}\n```")
            used += len(content)
        if file_blocks:
            parts.append("\n--- CURRENT FILE CONTENTS (complete, ready to edit) ---\n"
                         + "\n\n".join(file_blocks))
        if skipped:
            parts.append("Large files not inlined (ask if you specifically need them): "
                         + ", ".join(skipped))
    else:
        files = result.get("all_files", [])
        if files:
            parts.append("Generated files: " + ", ".join(os.path.basename(f) for f in files[:25]))
        parts.append("(The build has not finished yet — file contents unavailable.)")

    return "\n\n".join(parts)


async def chat_reply(job: dict, history: list[dict], user_message: str) -> str:
    """Generate a reply, applying any ===FILE:=== blocks to disk before returning."""
    context = _build_context(job)

    # The project context (full file contents) is large and stable between edits, but it
    # used to be re-sent inside every user message — re-billing ~30k input tokens per turn.
    # Put it in a cache_control'd system block instead: follow-up turns within the cache
    # window (~5 min) re-read it at ~10% input cost. A file edit invalidates the cache for
    # one turn (contents changed), then it re-caches — a miss just costs what it did before,
    # so this is pure upside for the usage limit.
    system = [
        {"type": "text", "text": SYSTEM},
        {"type": "text",
         "text": f"=== CURRENT PROJECT (complete file contents) ===\n{context}",
         "cache_control": {"type": "ephemeral"}},
    ]

    messages = []
    for turn in history[-10:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_message})

    # Stream (so large multi-file edits don't hit a request timeout) with thinking on for
    # higher-quality edits. We accumulate only the text deltas — thinking never reaches disk.
    # Sonnet 4.6 at medium effort handles targeted edits against fully-inlined context well
    # for a fraction of opus/high cost; the context is already cached (see above).
    full_text = ""
    async for kind, delta in stream(
        model=MODEL_CODE, max_tokens=16000, thinking=True, effort="medium",
        system=system, messages=messages,
    ):
        if kind == "text":
            full_text += delta
    raw = full_text

    # Apply any file edits the model produced.
    result = job.get("result") or {}
    selected = result.get("selected_idea")
    changed: list[str] = []
    if selected:
        project_title = selected["title"]
        for rel_path, content in parse_file_blocks(raw):
            write_file(project_title, rel_path, content)
            changed.append(rel_path)

    # Strip the ===FILE:===...===END=== blocks so the chat panel shows prose, not a wall of code.
    clean = _strip_file_blocks(raw).strip()

    if changed:
        change_list = "\n".join(f"- `{p}`" for p in changed)
        if clean:
            return f"{clean}\n\n**Files updated:**\n{change_list}"
        return f"Done. Updated {len(changed)} file(s):\n{change_list}"

    return clean or raw


def _strip_file_blocks(text: str) -> str:
    """Remove ===FILE:=== ... ===END=== blocks, leaving only prose."""
    import re
    return re.sub(r"===FILE:[^\n]*===.*?===END===", "", text, flags=re.DOTALL).strip()
