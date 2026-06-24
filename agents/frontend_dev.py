from typing import AsyncGenerator
from agents.llm import stream, MODEL_CODE
from tools.file_writer import write_file_blocks, get_output_path, list_output_files

SYSTEM = (
    "You are a senior frontend engineer on a hackathon team. You write COMPLETE, "
    "production-quality, immediately-runnable code — never placeholders, never TODOs, "
    "never '// rest of code here'. Every file you output must be whole and final. "
    "You are given a precise BUILD SPEC (the shared contract for the whole project) "
    "and you must follow its API endpoints, base URL, and file manifest EXACTLY so "
    "that the frontend you build actually talks to the backend your teammate builds."
)


class FrontendDevAgent:
    async def run(
        self,
        project_title: str,
        project_description: str,
        spec_text: str,
        instructions: str = "",
    ) -> AsyncGenerator[dict, None]:
        yield _status("frontend_dev", f"Studying the build spec and designing the frontend for '{project_title}'…")

        extra = (
            f"\n\nADDITIONAL TEAM INSTRUCTIONS (these override defaults — follow them closely):\n{instructions}"
            if instructions else ""
        )

        prompt = f"""Build the COMPLETE frontend for this hackathon project.

PROJECT: {project_title}
DESCRIPTION: {project_description}

══════════ SHARED BUILD SPEC (the single source of truth for the whole team) ══════════
{spec_text}
═══════════════════════════════════════════════════════════════════════════════════════
{extra}

Think carefully first, then produce every file. Requirements:

1. COMPLETENESS — produce EVERY file listed under the spec's frontend file manifest.
   At an absolute minimum: index.html, style.css, app.js. Do NOT reference a file
   (a <link>, <script src>, or fetch of a local asset) that you do not also create.
   A page that links a missing style.css or app.js is a FAILED deliverable.

2. WIRED TO THE BACKEND — call the backend using the EXACT endpoint paths, HTTP
   methods, and request/response shapes from the spec. At the very top of app.js
   define a single configurable base URL constant, e.g.:
       const API_BASE = window.API_BASE || "http://127.0.0.1:<backend_port from spec>";
   and build every request off it. Handle loading and error states (show the user a
   message if the backend is unreachable) so a demo never silently hangs.

3. SELF-CONTAINED & RUNNABLE — vanilla HTML/CSS/JS unless the spec's tech stack
   explicitly says React. No build step unless the spec demands one. It must work by
   opening index.html (or serving the folder) with the backend running.

4. POLISHED — a clean, modern, impressive UI judges will remember: real layout,
   thoughtful styling, responsive, with the project's actual features wired up — not
   a skeleton.

5. CLOSE YOUR DEPENDENCY GRAPH before you finish. A passing look is not enough — the
   #1 way these builds break is referencing something that was never created. Walk
   every reference and confirm it resolves:
   - Every <script src>, <link href>, <img src>, and other local asset in your HTML
     is a file you also output in this response.
   - Every ES-module `import ... from './x.js'` resolves to a file you also output.
     A single failed top-level import silently bricks the entire page.
   - Every fetch()/API call uses an exact path + method from the spec's API contract.
   - If your code needs a file or endpoint you are NOT creating, that is a problem —
     either create it here or call the right existing artifact. Do not reference a
     file "someone will add later."

6. FAIL LOUD, NOT SILENT. Prefer code where a missing/failed dependency shows a clear,
   visible error over code that silently hangs. Resolve every loading state into real
   content or an explicit error message — never leave the user staring at a spinner.

Output format — emit each file as a block, with NOTHING between blocks:
===FILE: index.html===
<full file contents>
===END===
===FILE: style.css===
<full file contents>
===END===

Use paths relative to the frontend/ directory (e.g. "index.html", "js/app.js").
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
            yield _thought("frontend_dev", delta)

        files_written = write_file_blocks(project_title, "frontend", full_text)
        yield _status("frontend_dev", f"Wrote {len(files_written)} frontend files.")
        yield _data("frontend_dev", {"files": files_written})

    async def repair(self, project_title: str, spec_text: str, failures: str) -> AsyncGenerator[dict, None]:
        """Fix-loop entry point: QA found frontend problems (e.g. referenced css/js
        that don't exist) — create/fix the missing files."""
        yield _status("frontend_dev", "QA reported frontend issues — creating the missing/broken files…")

        output_path = get_output_path(project_title)
        rel_files = [f for f in list_output_files(project_title) if f.startswith("frontend")]
        snippets = []
        for rel in rel_files[:10]:
            try:
                content = (output_path / rel).read_text(encoding="utf-8")[:5000]
                snippets.append(f"### {rel}\n{content}")
            except OSError:
                pass
        existing = "\n\n".join(snippets) or "No frontend files found."

        prompt = f"""You are the frontend engineer. QA found problems with the frontend.

QA failures:
{failures[:3000]}

Build spec (for the API contract / base URL):
{spec_text[:2000]}

Existing frontend files:
{existing}

Fix the problem completely. If index.html references a file that doesn't exist, CREATE that
file with full, working contents (do not just remove the reference unless it's truly unused).
Re-output every file you create or change, in full.

Output format — paths relative to the frontend/ directory:
===FILE: style.css===
<full contents>
===END===
Output ONLY file blocks."""

        full_text = ""
        async for kind, delta in stream(
            model=MODEL_CODE, max_tokens=24000, thinking=True, effort="high", system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        ):
            if kind == "text":
                full_text += delta
            yield _thought("frontend_dev", delta)

        fixed = write_file_blocks(project_title, "frontend", full_text)
        yield _status("frontend_dev", f"Created/fixed {len(fixed)} frontend file(s).")
        yield _data("frontend_dev", {"files": fixed})


def _status(agent: str, message: str) -> dict:
    return {"type": "status", "agent": agent, "message": message}

def _thought(agent: str, chunk: str) -> dict:
    return {"type": "thought", "agent": agent, "chunk": chunk}

def _data(agent: str, payload: dict) -> dict:
    return {"type": "data", "agent": agent, "data": payload}
