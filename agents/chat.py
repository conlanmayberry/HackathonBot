import os
import asyncio
import json
import anthropic

MODEL = "claude-opus-4-8"

SYSTEM = """You are the HackathonBot assistant — the coordinator of an AI team that builds hackathon projects.
You help the user refine their project with additional instructions, answer questions about the generated
work, and suggest improvements. You have full context on the current project below. Be concise and practical.
If the user gives an instruction that would require regenerating code (e.g. "rewrite the backend in Flask"),
tell them to add it to the instructions and re-run the team, and summarize what you would change."""


def _build_context(job: dict) -> str:
    """Summarize the current job state for the assistant."""
    req = job.get("request", {})
    result = job.get("result") or {}

    parts = [
        f"Hackathon: {req.get('hackathon', 'N/A')}",
        f"University: {req.get('university', 'N/A')}",
        f"Theme: {req.get('theme', 'N/A')}",
    ]

    selected = result.get("selected_idea")
    if selected:
        parts.append(f"Selected idea: {selected.get('title')} — {selected.get('description', '')}")
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
        parts.append(f"Run: backend on port {run.get('backend_port')}, frontend on port {run.get('frontend_port')}.")

    files = result.get("all_files", [])
    if files:
        parts.append(f"Generated files ({len(files)}): " + ", ".join(os.path.basename(f) for f in files[:25]))

    debug = result.get("debug", {})
    if debug.get("issues"):
        parts.append(f"Debugger findings:\n{debug['issues'][:800]}")

    if not selected:
        parts.append("(The team has not finished running yet, so some context may be missing.)")

    return "\n\n".join(parts)


async def chat_reply(job: dict, history: list[dict], user_message: str) -> str:
    """Generate an assistant reply given job context and prior conversation."""
    context = _build_context(job)

    messages = []
    for turn in history[-10:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({
        "role": "user",
        "content": f"[Current project context]\n{context}\n\n[User message]\n{user_message}",
    })

    client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), timeout=120.0, max_retries=2)
    resp = await client.messages.create(
        model=MODEL,
        max_tokens=1200,
        system=SYSTEM,
        messages=messages,
    )
    return resp.content[0].text
