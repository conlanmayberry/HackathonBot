from typing import AsyncGenerator
from agents.llm import stream, MODEL_CODE
from tools.file_writer import write_file_blocks, get_output_path, list_output_files

SYSTEM = (
    "You are a senior backend engineer on a hackathon team. You write COMPLETE, "
    "production-quality, immediately-runnable Python — never placeholders, never TODOs, "
    "never stubs. Every file you output is whole and final. You implement the shared "
    "BUILD SPEC's API contract EXACTLY (same paths, methods, request/response shapes) so "
    "the frontend your teammate builds works against your server without changes. You are "
    "fanatical about dependency hygiene: every third-party package you import appears in "
    "requirements.txt, and every secret you read appears in .env.example."
)


class BackendDevAgent:
    async def run(
        self,
        project_title: str,
        project_description: str,
        spec_text: str,
        instructions: str = "",
    ) -> AsyncGenerator[dict, None]:
        yield _status("backend_dev", f"Studying the build spec and implementing the backend for '{project_title}'…")

        extra = (
            f"\n\nADDITIONAL TEAM INSTRUCTIONS (these override defaults — follow them closely):\n{instructions}"
            if instructions else ""
        )

        prompt = f"""Build the COMPLETE backend for this hackathon project.

PROJECT: {project_title}
DESCRIPTION: {project_description}

══════════ SHARED BUILD SPEC (the single source of truth for the whole team) ══════════
{spec_text}
═══════════════════════════════════════════════════════════════════════════════════════
{extra}

Think carefully first, then produce every file. Requirements:

1. STACK — Python + FastAPI. Add CORS middleware allowing all origins/methods/headers
   (the frontend is served from a different port). Run via `uvicorn main:app`.

2. IMPLEMENT THE CONTRACT EXACTLY — create every endpoint in the spec with the exact
   path, method, and request/response JSON shape. Use Pydantic models for request
   bodies and validation. Return correct status codes (404 for missing resources,
   422 handled by FastAPI, etc.). Write REAL working logic — no fake handlers.

3. DEPENDENCIES — CLEAR AND RELIABLE. requirements.txt MUST list EVERY third-party
   package you import, one per line, with a `>=` lower bound, e.g.:
       fastapi>=0.115.0
       uvicorn[standard]>=0.32.0
       pydantic>=2.0.0
       python-dotenv>=1.0.0
   If your code imports it, it is in requirements.txt. If it is only in the standard
   library (json, os, uuid, datetime, asyncio, sqlite3, …) it is NOT in requirements.txt.

4. LLM USAGE — if this project needs an AI/LLM call, use the Anthropic API
   (`anthropic` package, model "claude-opus-4-8") reading ANTHROPIC_API_KEY from the
   environment. Do NOT use OpenAI/GPT — the team's environment only has an Anthropic key.

5. SAFE TO IMPORT — never make a network call, open a file, or require a real secret at
   import time. Read env vars with os.getenv(NAME, "") and only call external services
   inside route handlers. The test suite imports this module directly, so importing
   `main` with no API keys set must never raise.

6. CONFIG & DOCS — provide:
   - .env.example listing EVERY environment variable the app reads, with a short comment.
   - README.md: what it is, how to install (`pip install -r requirements.txt`), how to
     run (`uvicorn main:app --host 127.0.0.1 --port <backend_port from spec>`), and a
     list of every endpoint with an example curl. Use in-memory storage if a database
     would be overkill (and say so in the README).

7. CLOSE YOUR DEPENDENCY GRAPH before you finish. The frontend will call the endpoints
   in the spec's API contract EXACTLY as written — implement every one of them with the
   matching path, method, and response shape. A frontend call to a route you didn't
   create is a broken product. Likewise: every module you `import` resolves to the
   standard library, a package in requirements.txt, or a file you also output here.

8. FAIL LOUD, NOT SILENT. Return clear error responses (correct status codes + a JSON
   body) rather than letting a handler raise an unhandled 500. Validate at the boundary.

Output format — emit each file as a block, with NOTHING between blocks:
===FILE: main.py===
<full file contents>
===END===
===FILE: requirements.txt===
<full file contents>
===END===

Use paths relative to the backend/ directory (e.g. "main.py", "models.py").
At minimum produce: main.py, requirements.txt, .env.example, README.md.
Output ONLY file blocks (a short plan beforehand is fine, but every artifact must be
inside a ===FILE: ...=== block)."""

        full_text = ""
        async for kind, delta in stream(
            model=MODEL_CODE,
            max_tokens=32000,
            thinking=True,
            effort="high",
            system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        ):
            if kind == "text":
                full_text += delta
            yield _thought("backend_dev", delta)

        files_written = write_file_blocks(project_title, "backend", full_text)
        yield _status("backend_dev", f"Wrote {len(files_written)} backend files.")
        yield _data("backend_dev", {"files": files_written})

    async def repair(self, project_title: str, spec_text: str, failures: str) -> AsyncGenerator[dict, None]:
        """Fix-loop entry point: QA found failures — rewrite the broken files.
        Can touch backend/* and the test file, using project-root-relative paths."""
        yield _status("backend_dev", "QA reported failures — reviewing the errors and fixing the code…")

        output_path = get_output_path(project_title)
        rel_files = [f for f in list_output_files(project_title)
                     if f.endswith(".py") and (f.startswith("backend") or f.startswith("tests"))]
        snippets = []
        for rel in rel_files[:12]:
            try:
                content = (output_path / rel).read_text(encoding="utf-8")[:5000]
                snippets.append(f"===FILE: {rel.replace(chr(92), '/')}===\n{content}\n===END===")
            except OSError:
                pass
        code_context = "\n\n".join(snippets) or "No code found."

        prompt = f"""You are the backend engineer. QA ran pylint + pytest and found problems.
Fix them so pylint has no errors and pytest passes. Make the SMALLEST changes that work.

QA failures:
{failures[:4000]}

The build spec (the contract you must keep matching):
{spec_text[:2500]}

Current files (paths are relative to the project root):
{code_context}

Decide whether the bug is in the application code or in a test, and fix the right file(s).
Output the ENTIRE corrected contents of every file you change as a block, using the SAME
project-root-relative path shown above (e.g. "backend/main.py" or "tests/test_backend.py").
Only re-output files you actually change. Output ONLY file blocks."""

        full_text = ""
        async for kind, delta in stream(
            model=MODEL_CODE, max_tokens=24000, thinking=True, effort="high", system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        ):
            if kind == "text":
                full_text += delta
            yield _thought("backend_dev", delta)

        # Paths are project-root-relative, so write with no subdir prefix.
        fixed = write_file_blocks(project_title, "", full_text)
        yield _status("backend_dev", f"Applied fixes to {len(fixed)} file(s).")
        yield _data("backend_dev", {"files": fixed})


def _status(agent: str, message: str) -> dict:
    return {"type": "status", "agent": agent, "message": message}

def _thought(agent: str, chunk: str) -> dict:
    return {"type": "thought", "agent": agent, "chunk": chunk}

def _data(agent: str, payload: dict) -> dict:
    return {"type": "data", "agent": agent, "data": payload}
