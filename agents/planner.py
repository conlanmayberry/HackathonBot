import json
import asyncio
from typing import AsyncGenerator

from agents.llm import stream, MODEL, MODEL_CODE
from tools.devpost import search_devpost_winners
from tools.file_writer import list_output_files, get_output_path, write_file
from tools.memory import recall_relevant, summarize_lessons, record_build
from agents.frontend_dev import FrontendDevAgent
from agents.backend_dev import BackendDevAgent
from agents.qa import QAAgent

DEFAULT_BACKEND_PORT = 8100   # matches tools/runner.py preset_commands
DEFAULT_FRONTEND_PORT = 8200
MAX_FIX_ROUNDS = 2            # build → verify → fix, capped so it can't loop forever


class PlannerAgent:
    """The lead agent. Absorbs the former Supervisor (orchestration) and Researcher
    (Devpost analysis) roles, and additionally acts as the project ARCHITECT — it
    writes a single shared build spec that every downstream agent builds against, and
    the INTEGRATOR — it writes the root-level glue files (README, .env.example,
    .gitignore, run scripts) that tie the frontend and backend into one runnable app."""

    def __init__(self):
        self.frontend_dev = FrontendDevAgent()
        self.backend_dev = BackendDevAgent()
        self.qa = QAAgent()

    async def run(
        self,
        hackathon: str,
        university: str,
        theme: str,
        autonomous: bool = False,
        instructions: str = "",
        select_idea=None,
        ask_user=None,
    ) -> AsyncGenerator[dict, None]:

        yield _status("planner", f"Starting HackathonBot for {hackathon} at {university}")
        if instructions:
            yield _status("planner", f"Applying additional instructions: \"{instructions[:120]}\"")

        # ── Step 1: Research past winners on Devpost ────────────────────────
        yield _status("planner", f"Researching past '{theme}' winners on Devpost…")
        try:
            winners = await search_devpost_winners(theme, max_results=15)
        except Exception:  # noqa: BLE001 — research must never block building
            winners = []
        winners_text = "\n".join(
            f"- {p['title']}: {p['description'][:200]}" for p in winners
        ) or "No results found — use general hackathon knowledge."
        yield _status("planner", f"Analyzed {len(winners)} past winners. Generating ideas…")

        # ── Step 2: Generate ideas + wait for selection (loops on regenerate) ─
        selected = {"title": theme, "description": theme, "tech_stack": ["Python", "JavaScript", "FastAPI"]}
        async for item in self._select_idea_loop(
            hackathon, university, theme, instructions, winners_text, len(winners),
            autonomous, select_idea,
        ):
            if isinstance(item, dict) and item.get("__selected__"):
                selected = item["idea"]
            else:
                yield item

        # ── Step 2.5: KICKOFF CHAT — ask the user how to proceed (optional) ─
        # Only when interactive (not autonomous) and a chat channel is wired up.
        if ask_user is not None and not autonomous:
            async for item in self._kickoff_chat(selected, instructions, ask_user):
                if isinstance(item, dict) and item.get("__instructions__"):
                    instructions = item["instructions"]
                else:
                    yield item

        # ── Step 3: ARCHITECT — design the shared build spec ────────────────
        # Recall lessons from similar past builds so the architect can pre-empt the
        # failures that drove fix-loop rounds before (advisory only — empty if no history).
        lessons = summarize_lessons(recall_relevant(theme, selected.get("tech_stack", [])))
        if lessons:
            yield _status("planner", "Recalling lessons from past builds to harden the spec…", data={"lessons": lessons})

        yield _status("planner", "Architecting the build spec — defining the API contract, file manifest, env vars and ports…")
        spec, spec_text = {}, ""
        async for item in self._architect_spec(selected, instructions, lessons):
            if isinstance(item, dict) and item.get("__spec__"):
                spec, spec_text = item["spec"], item["spec_text"]
            else:
                yield item
        yield _status("planner", "Build spec ready — frontend and backend will build against the same contract.", data={"report": spec_text})
        yield _data("planner", {"report": spec_text})

        backend_port = _port(spec, "backend", DEFAULT_BACKEND_PORT)
        frontend_port = _port(spec, "frontend", DEFAULT_FRONTEND_PORT)

        # ── Step 4: Frontend + Backend in parallel, both fed the same spec ──
        yield _status("planner", "Launching Frontend Dev and Backend Dev in parallel — both working from the shared spec…")

        # The root README is written from the spec, which is already final — so kick it off
        # NOW and let it generate concurrently with the (much longer) dev phase instead of
        # serializing it afterwards. We await the result at the integration step below.
        readme_task = asyncio.create_task(
            self._write_root_readme(selected, spec_text, backend_port, frontend_port)
        )

        q: asyncio.Queue = asyncio.Queue()
        fe_result: dict = {}
        be_result: dict = {}

        async def drain(gen, sentinel):
            async for ev in gen:
                await q.put(ev)
            await q.put(sentinel)

        asyncio.create_task(drain(
            _safe_stage(self.frontend_dev.run(selected["title"], selected.get("description", ""), spec_text, instructions), "frontend_dev"),
            {"type": "__done__", "agent": "frontend_dev"},
        ))
        asyncio.create_task(drain(
            _safe_stage(self.backend_dev.run(selected["title"], selected.get("description", ""), spec_text, instructions), "backend_dev"),
            {"type": "__done__", "agent": "backend_dev"},
        ))

        done_count = 0
        while done_count < 2:
            event = await q.get()
            if event["type"] == "__done__":
                done_count += 1
                continue
            yield event
            if event["type"] == "data":
                if event["agent"] == "frontend_dev":
                    fe_result = event["data"]
                elif event["agent"] == "backend_dev":
                    be_result = event["data"]

        yield _status("frontend_dev", f"Frontend done — {len(fe_result.get('files', []))} files written", data={"files": fe_result.get("files", [])})
        yield _status("backend_dev", f"Backend done — {len(be_result.get('files', []))} files written", data={"files": be_result.get("files", [])})

        # ── Step 5: INTEGRATE — write the root-level glue files ─────────────
        yield _status("planner", "Writing root-level integration files (README, .env.example, .gitignore, run scripts)…")
        glue_files = await self._write_integration_files(selected, spec, spec_text, backend_port, frontend_port, readme_task)
        yield _status("planner", f"Wrote {len(glue_files)} integration files so the project installs and runs as one app.", data={"files": glue_files})

        # ── Step 6: QA + fix loop (build → verify → devs fix → re-verify) ───
        async for event in _safe_stage(self.qa.setup(selected["title"], spec_text), "qa"):
            yield event

        qa_result: dict = {}
        for attempt in range(MAX_FIX_ROUNDS + 1):
            async for event in _safe_stage(self.qa.verify(selected["title"]), "qa"):
                yield event
                if event["type"] == "data":
                    qa_result = event["data"]
                    if event["data"].get("issues"):
                        # surface to the Debug/Tests panel
                        yield _status("qa", "QA checks complete.", data={"issues": event["data"]["issues"], "pytest_output": event["data"].get("pytest_output", "")})

            if qa_result.get("passed"):
                yield _status("qa", "✅ All QA checks pass — the project builds, lints clean and tests pass.")
                break
            if attempt == MAX_FIX_ROUNDS:
                yield _status("qa", "⚠️ Some checks still failing after fixes — see the Debug/Tests panel for details.")
                break

            yield _status("planner", f"QA found issues — handing them back to the devs to fix (round {attempt + 1} of {MAX_FIX_ROUNDS})…")
            if qa_result.get("backend_failures"):
                async for event in _safe_stage(self.backend_dev.repair(selected["title"], spec_text, qa_result["backend_failures"]), "backend_dev"):
                    yield event
            if qa_result.get("frontend_failures"):
                async for event in _safe_stage(self.frontend_dev.repair(selected["title"], spec_text, qa_result["frontend_failures"]), "frontend_dev"):
                    yield event

        # Remember this build's outcome (theme, stack, QA result, fix-round count) so future
        # architect runs can recall it. ``attempt`` holds the rounds used; best-effort, never raises.
        record_build(theme=theme, hackathon=hackathon, idea=selected, spec=spec,
                     qa_result=qa_result, fix_rounds=attempt)

        # ── Step 7: INSTALL GUIDE — write the explicit setup instructions LAST, now that
        #    requirements.txt is final (QA reconciled it during the fix loop). Covers the
        #    system-level prerequisites the architect declared (game engine, runtimes, DB
        #    servers, …) AND enumerates every backend package the "Install backend deps"
        #    quick action installs, so the Files tab shows exactly what to set up. ───────
        yield _status("planner", "Writing INSTALL.md — explicit setup steps (system prerequisites + every backend dependency)…")
        install_guide = await self._write_install_guide(selected, spec, backend_port, frontend_port)
        write_file(selected["title"], "INSTALL.md", install_guide)
        yield _status("planner", "Wrote INSTALL.md — step-by-step install instructions are in the Files tab.", data={"files": ["INSTALL.md"]})

        # ── Final summary ───────────────────────────────────────────────────
        all_files = list_output_files(selected["title"])
        output_path = str(get_output_path(selected["title"]))

        final_result = {
            "selected_idea": selected,
            "build_spec": spec,
            "frontend": fe_result,
            "backend": be_result,
            "qa": qa_result,
            # Back-compat shapes for the existing Debug/Tests panel:
            "debug": {"issues": qa_result.get("issues", ""), "files_checked": qa_result.get("files_checked", 0)},
            "tests": {"pytest_output": qa_result.get("pytest_output", "")},
            "all_files": all_files,
            "output_path": output_path,
            "run": {"backend_port": backend_port, "frontend_port": frontend_port},
            "install_guide": install_guide,
        }

        yield _status(
            "planner",
            f"Done! '{selected['title']}' scaffolded at {output_path} ({len(all_files)} files). "
            f"See README.md for install & run steps (backend :{backend_port}, frontend :{frontend_port}).",
            data={"result": final_result},
        )

    # ── Idea generation + selection ─────────────────────────────────────────
    async def _select_idea_loop(self, hackathon, university, theme, instructions,
                                 winners_text, winners_count, autonomous, select_idea):
        """Async generator. Yields UI events, and finally yields a sentinel dict
        {"__selected__": True, "idea": {...}} carrying the chosen idea."""
        seen_titles: list[str] = []
        while True:
            avoid = seen_titles or None

            extra = (
                f"\n\nIMPORTANT — additional instructions from the team (follow these closely):\n{instructions}"
                if instructions else ""
            )
            avoid_block = ""
            if avoid:
                avoid_block = (
                    "\n\nThe team has already seen and rejected these ideas — do NOT repeat or lightly "
                    f"reword them. Produce clearly DIFFERENT concepts:\n{'; '.join(avoid)}"
                )

            # Stable context (role, theme, winners research, team instructions) goes in the
            # system prompt and is cached; only the volatile "avoid these" list changes
            # between rounds, so a "regenerate" reuses the cached Opus prefix.
            system_ctx = f"""You are a hackathon strategist helping a college team win {hackathon} at {university}.

Theme/category: {theme}

Past winning projects found on Devpost:
{winners_text}{extra}"""

            prompt = f"""Generate 3-5 project ideas for the team. About 60% should build on patterns
you see in the past winners above, 40% should be fresh/original. Each idea must be buildable in a
hackathon as a web app (JS frontend + Python/FastAPI backend).{avoid_block}

For each idea return a JSON object with:
- title: short project name
- description: 2-3 sentence pitch
- tech_stack: list of recommended technologies
- why_it_wins: 1-2 sentences on why judges would pick this
- originality: "inspired" or "original"

Return a JSON array of these objects. No markdown fences, just raw JSON."""

            full_text = ""
            async for kind, delta in stream(
                model=MODEL,
                max_tokens=4000,
                system=system_ctx,
                cache=True,
                messages=[{"role": "user", "content": prompt}],
            ):
                if kind == "text":
                    full_text += delta
                yield _thought("planner", delta)

            ideas = _parse_json_array(full_text)
            seen_titles.extend(i.get("title", "") for i in ideas)
            yield _status("planner", f"Generated {len(ideas)} ideas ({winners_count} past winners analyzed)", data={"ideas": ideas})
            yield _data("planner", {"ideas": ideas, "winners_researched": winners_count})

            if not ideas:
                yield {"__selected__": True, "idea": {"title": theme, "description": theme, "tech_stack": ["Python", "JavaScript", "FastAPI"]}}
                return

            if autonomous or select_idea is None:
                yield _status("planner", f"Auto-selected: '{ideas[0]['title']}'", data={"selection_made": True})
                yield {"__selected__": True, "idea": ideas[0]}
                return

            yield _status("planner", "Waiting for you to choose an idea, add your own, or regenerate…", data={"awaiting_selection": True})
            choice = await select_idea() or {}
            if choice.get("action") == "regenerate":
                yield _status("planner", "Regenerating a fresh set of ideas…", data={"regenerating": True})
                continue

            custom = choice.get("custom_idea")
            if custom and custom.get("title"):
                idea = {
                    "title": custom.get("title"),
                    "description": custom.get("description", ""),
                    "tech_stack": custom.get("tech_stack") or ["Python", "JavaScript", "FastAPI"],
                    "why_it_wins": custom.get("why_it_wins", ""),
                    "originality": "original",
                }
                yield _status("planner", f"Building your own idea: '{idea['title']}'.", data={"selection_made": True})
                yield {"__selected__": True, "idea": idea}
                return

            idx = choice.get("index")
            idx = idx if isinstance(idx, int) and 0 <= idx < len(ideas) else 0
            yield _status("planner", f"You selected: '{ideas[idx]['title']}' — building it now.", data={"selection_made": True})
            yield {"__selected__": True, "idea": ideas[idx]}
            return

    # ── Kickoff chat: ask the user how they'd like to proceed ───────────────
    async def _kickoff_chat(self, selected, instructions, ask_user):
        """Optionally ask the user a few clarifying questions in the Chat tab before
        building. Non-blocking: if they don't answer (timeout), we proceed with our
        best judgment. Yields events, then a sentinel {"__instructions__": ...}."""
        extra = f"\nThe team already gave these instructions: {instructions}" if instructions else ""
        prompt = f"""You are the planner about to build this hackathon project:

Title: {selected.get('title')}
Description: {selected.get('description', '')}
Tech stack: {', '.join(selected.get('tech_stack', []) or [])}{extra}

Before building, what would you most want to confirm with the team? Produce up to 3 SHORT,
high-value clarifying questions whose answers would genuinely change what you build — e.g.
visual style/branding, whether to prioritize a polished demo flow vs. breadth of features,
must-have features or integrations, or the target user. If the idea is already clear enough
to build well without asking, return an empty array.

Return ONLY a JSON array of question strings (e.g. ["Q1?", "Q2?"]). No prose."""

        # Short, well-scoped clarifying questions — cheap boilerplate, runs on Sonnet.
        full_text = ""
        async for kind, delta in stream(
            model=MODEL_CODE, max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        ):
            if kind == "text":
                full_text += delta

        questions = _parse_json_array(full_text)
        questions = [q for q in questions if isinstance(q, str) and q.strip()][:3]
        if not questions:
            yield {"__instructions__": True, "instructions": instructions}
            return

        qtext = (
            "Before I start building, a few quick questions so I build what you actually want:\n\n"
            + "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
            + "\n\nReply here with your preferences — or just say \"go ahead\" and I'll use my best judgment."
        )
        yield _status("planner", "Asking you a few clarifying questions in the Chat tab — answer there, or I'll proceed shortly…", data={"awaiting_chat": True})

        answer = await ask_user(qtext)  # posts the question to chat and waits (with timeout)

        if answer and answer.strip() and answer.strip().lower() not in ("go ahead", "go", "proceed", "skip", "no"):
            new_instructions = (instructions + "\n\n" if instructions else "") + (
                f"Team preferences captured from the kickoff chat:\nQuestions: {'; '.join(questions)}\nAnswer: {answer.strip()}"
            )
            yield _chat("planner", "Got it — thanks! I'll build with your preferences in mind. 🚀")
            yield {"__instructions__": True, "instructions": new_instructions}
        else:
            yield _chat("planner", "No problem — I'll use my best judgment and get started. 🚀")
            yield {"__instructions__": True, "instructions": instructions}

    # ── Architect the shared build spec ─────────────────────────────────────
    async def _architect_spec(self, selected, instructions, lessons=""):
        """Design the contract every agent builds against. Async generator: yields
        live thought events, then a sentinel {"__spec__": True, "spec", "spec_text"}.
        ``lessons`` (optional) is advisory text recalled from similar past builds."""
        tech = ", ".join(selected.get("tech_stack", []) or ["Python", "FastAPI", "JavaScript"])
        extra = f"\n\nAdditional team instructions to honor:\n{instructions}" if instructions else ""
        lessons_block = (
            f"\n\nLESSONS FROM PAST BUILDS (apply these to avoid repeating prior QA failures):\n{lessons}"
            if lessons else ""
        )

        prompt = f"""You are the technical architect for a hackathon team. Design the COMPLETE
build spec for this project so a frontend dev and a backend dev can work in parallel and
have their pieces fit together perfectly on the first try.

PROJECT: {selected.get('title')}
DESCRIPTION: {selected.get('description', '')}
SUGGESTED STACK: {tech}{extra}{lessons_block}

Think hard about the data model and the exact HTTP contract, then output ONLY a JSON
object (no markdown fences) with this shape:

{{
  "slug": "kebab-case-name",
  "summary": "one-paragraph description of what gets built",
  "tech_stack": {{"frontend": ["..."], "backend": ["FastAPI", "..."]}},
  "system_requirements": [{{"name": "Node.js", "version": "18+", "why": "what it's needed for", "install": "https://nodejs.org  (or `brew install node` / `winget install OpenJS.NodeJS`)"}}],
  "ports": {{"backend": {DEFAULT_BACKEND_PORT}, "frontend": {DEFAULT_FRONTEND_PORT}}},
  "env_vars": [{{"name": "ANTHROPIC_API_KEY", "description": "...", "required": true}}],
  "data_models": [{{"name": "Item", "fields": {{"id": "str", "name": "str"}}}}],
  "api_endpoints": [
    {{"method": "POST", "path": "/items", "description": "create an item",
      "request": {{"name": "str"}}, "response": {{"id": "str", "name": "str"}}}}
  ],
  "file_manifest": {{
    "frontend": [{{"path": "index.html", "purpose": "..."}}, {{"path": "style.css", "purpose": "..."}}, {{"path": "app.js", "purpose": "..."}}],
    "backend": [{{"path": "main.py", "purpose": "..."}}, {{"path": "requirements.txt", "purpose": "..."}}, {{"path": ".env.example", "purpose": "..."}}, {{"path": "README.md", "purpose": "..."}}]
  }},
  "integration_notes": "exactly how the frontend calls the backend: base URL, CORS, request/response examples",
  "frontend_notes": "key UI screens and behaviors",
  "backend_notes": "key logic, storage approach, any LLM usage (use Anthropic claude-opus-4-8 if needed)"
}}

CRITICAL — the manifest is the ownership contract that prevents integration gaps (the #1
multi-agent failure: one agent references an artifact another agent never built):
- The file_manifest must list EVERY file each side needs to run — including every JS
  module that another file imports, every stylesheet/asset the HTML links, and every
  backend module that main.py imports. If a file will be referenced anywhere, it MUST
  appear in the manifest with an owner (its frontend/backend section). Nothing may be
  referenced without an owner.
- api_endpoints is the frontend↔backend contract: list EVERY endpoint the frontend will
  call, concrete and complete. The frontend calls exactly these (path + method); the
  backend implements exactly these. No feature may need an endpoint that isn't listed.
- system_requirements lists EVERY non-pip, system-level prerequisite a developer must
  install BY HAND before the app runs: game engines (Unity, Godot, Unreal), language
  runtimes (Node.js, Go, .NET), database/cache servers (PostgreSQL, Redis, MongoDB), and
  native CLI tools (ffmpeg, Docker, Tesseract, etc.) — each with a concrete version and
  how to install it (a download link and/or package-manager commands for macOS/Windows/
  Linux). Be specific: name the exact engine and edition the project assumes. Python
  packages do NOT go here — they belong in requirements.txt. If the project is pure Python
  plus a static HTML/CSS/JS frontend (no extra software needed), use an empty array [].
Use realistic field names. If an LLM is needed, specify Anthropic (ANTHROPIC_API_KEY)."""

        full_text = ""
        async for kind, delta in stream(
            model=MODEL,
            max_tokens=20000,
            thinking=True,
            effort="high",
            messages=[{"role": "user", "content": prompt}],
        ):
            if kind == "text":
                full_text += delta
            yield _thought("planner", delta)

        spec = _parse_json_object(full_text)
        if not spec:
            # Fall back: still give the devs the raw guidance as a contract.
            spec = {
                "slug": "",
                "summary": selected.get("description", ""),
                "tech_stack": {"frontend": ["HTML", "CSS", "JavaScript"], "backend": selected.get("tech_stack", ["FastAPI"])},
                "ports": {"backend": DEFAULT_BACKEND_PORT, "frontend": DEFAULT_FRONTEND_PORT},
                "env_vars": [{"name": "ANTHROPIC_API_KEY", "description": "Anthropic API key", "required": True}],
                "api_endpoints": [],
                "file_manifest": {},
            }
            spec["_raw"] = full_text.strip()
        yield {"__spec__": True, "spec": spec, "spec_text": _format_spec(spec, selected)}

    # ── Integration / glue files ────────────────────────────────────────────
    async def _write_integration_files(self, selected, spec, spec_text, backend_port, frontend_port, readme_task=None) -> list[str]:
        title = selected["title"]
        written = []

        # 1) Root .gitignore (deterministic, reliable).
        written.append(write_file(title, ".gitignore", _GITIGNORE))

        # 2) Root .env.example aggregated from the spec's env vars (deterministic).
        env_lines = ["# Copy this file to .env and fill in real values.\n"]
        names = set()
        for var in (spec.get("env_vars") or []):
            name = (var.get("name") or "").strip()
            if name and name not in names:
                names.add(name)
                desc = var.get("description", "")
                if desc:
                    env_lines.append(f"# {desc}")
                env_lines.append(f"{name}=")
        if "ANTHROPIC_API_KEY" not in names:
            env_lines.append("# Anthropic API key (used if the backend calls an LLM)")
            env_lines.append("ANTHROPIC_API_KEY=")
        written.append(write_file(title, ".env.example", "\n".join(env_lines) + "\n"))

        # 3) Cross-platform run scripts (deterministic — this is the reliable install path).
        written.append(write_file(title, "run.sh", _run_sh(backend_port, frontend_port)))
        written.append(write_file(title, "run.ps1", _run_ps1(backend_port, frontend_port)))

        # 4) Root README — LLM-written from the spec. It was kicked off back when the dev
        #    phase started (it only needs the spec), so by now it's usually already done;
        #    await the in-flight task instead of generating it from scratch here.
        if readme_task is not None:
            readme = await readme_task
        else:
            readme = await self._write_root_readme(selected, spec_text, backend_port, frontend_port)
        written.append(write_file(title, "README.md", readme))
        return written

    async def _write_root_readme(self, selected, spec_text, backend_port, frontend_port) -> str:
        prompt = f"""Write the ROOT README.md for this hackathon project. It must let a teammate go
from a fresh clone to a running demo with zero guesswork.

PROJECT: {selected.get('title')}
DESCRIPTION: {selected.get('description', '')}

Build spec:
{spec_text}

The repo layout is:
  backend/    FastAPI app (run with uvicorn on port {backend_port})
  frontend/   static HTML/CSS/JS (served on port {frontend_port})
  tests/      pytest suite
  run.sh / run.ps1   one-command setup+run scripts

Include, with real copy-pasteable commands:
1. What it is (2-3 sentences) and a short feature list.
2. Prerequisites (Python 3.10+).
3. Setup — create a venv, `pip install -r backend/requirements.txt`, copy `.env.example`
   to `.env` and fill in keys.
4. Run — the exact commands to start the backend (uvicorn, port {backend_port}) and to
   serve the frontend (`python -m http.server {frontend_port}` from frontend/), plus the
   one-liner `./run.sh` (mac/linux) / `./run.ps1` (windows) alternative.
5. Running tests (`pytest tests -v`).
6. An API reference table from the spec's endpoints.
7. Troubleshooting: missing deps, missing API key, CORS, port already in use.

Output GitHub-flavored markdown only."""
        # README is glue prose generated from the finished spec — Sonnet handles it well.
        # Runs concurrently with the dev phase, so guard it: a transient API error here must
        # degrade to a minimal README, never abort the build at the await point downstream.
        full_text = ""
        try:
            async for kind, delta in stream(
                model=MODEL_CODE,
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}],
            ):
                if kind == "text":
                    full_text += delta
        except Exception:  # noqa: BLE001 — README is non-critical glue; fall back gracefully
            import traceback
            traceback.print_exc()
        return full_text.strip() or f"# {selected.get('title')}\n\n{selected.get('description', '')}\n"

    async def _write_install_guide(self, selected, spec, backend_port, frontend_port) -> str:
        """Explicit, project-specific setup instructions — saved as INSTALL.md and shown in
        the Files tab. Covers (1) the system-level prerequisites the architect declared
        (game engines, runtimes, DB servers, native tools) and (2) EVERY backend Python
        package the '📦 Install backend deps' quick action installs, enumerated from the
        FINAL requirements.txt (QA has reconciled it by the time this runs)."""
        title = selected.get("title", "")
        output_path = get_output_path(title)

        # Deterministic: read the final requirements.txt so the dependency list is exactly
        # what `pip install -r backend/requirements.txt` (the quick action) pulls in.
        req_lines: list[str] = []
        try:
            for raw in (output_path / "backend" / "requirements.txt").read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if line and not line.startswith("#"):
                    req_lines.append(line)
        except OSError:
            pass
        reqs_block = "\n".join(req_lines) or "(no backend/requirements.txt found)"

        sys_lines = []
        for r in (spec.get("system_requirements") or []):
            if isinstance(r, dict):
                sys_lines.append(
                    f"- {r.get('name', '?')} {r.get('version', '')} — {r.get('why', '')} "
                    f"(install: {r.get('install', 'see official site')})"
                )
        sys_block = "\n".join(sys_lines) or "(none declared — pure Python + a static frontend)"
        ts = spec.get("tech_stack", {})
        tech_str = ", ".join((ts.get("frontend") or []) + (ts.get("backend") or [])) or "Python, FastAPI, HTML/CSS/JS"

        prompt = f"""Write INSTALL.md — explicit, copy-pasteable setup instructions for this hackathon
project. A teammate on a fresh machine must follow it top to bottom and end up with the app
running. Be SPECIFIC and concrete — no vague "install the dependencies" hand-waving.

PROJECT: {title}
TECH STACK: {tech_str}

SYSTEM PREREQUISITES the architect declared (non-pip software that must be installed by hand —
game engines, runtimes, database servers, native CLI tools):
{sys_block}

EXACT contents of backend/requirements.txt (this is precisely what gets installed):
{reqs_block}

Repo layout: backend/ (FastAPI, port {backend_port}), frontend/ (static, port {frontend_port}),
tests/ (pytest). There are run.sh / run.ps1 one-command scripts at the repo root too.

Write the guide as GitHub-flavored markdown with these sections:

1. "## 1. System prerequisites" — for EACH item above, a short sub-section giving the exact
   version and the real install command(s) for macOS / Windows / Linux (or a download link).
   Be concrete and name the exact engine/edition. If the list is empty, say plainly that the
   only requirement is Python 3.10+ and no extra software is needed.
2. "## 2. Python & backend dependencies" — create+activate a venv, then state that
   `pip install -r backend/requirements.txt` installs the packages. IMPORTANT: say explicitly
   that this is the SAME thing the "📦 Install backend deps" quick action in the app's Run tab
   does, so the user knows the button and this command are identical. Then list EVERY package
   from the requirements.txt above in a markdown table with two columns — Package and "What
   it's for" — and wrap each package/version in backticks (e.g. `fastapi>=0.115.0`). Do not
   omit any and do not invent any that aren't listed.
3. "## 3. Configuration" — copy .env.example to .env and fill in keys (call out ANTHROPIC_API_KEY
   if it appears in the deps/spec).
4. "## 4. Run it" — exact commands to start the backend (uvicorn on port {backend_port}) and the
   frontend (python -m http.server {frontend_port} from frontend/), plus the run.sh / run.ps1
   one-liner alternative.
5. "## 5. Verify it works" — open the frontend URL, and run `pytest tests -v`.

Output ONLY the markdown for INSTALL.md (no fences around the whole thing)."""

        full_text = ""
        async for kind, delta in stream(
            model=MODEL_CODE, max_tokens=4000,
            messages=[{"role": "user", "content": prompt}],
        ):
            if kind == "text":
                full_text += delta
        return full_text.strip() or _fallback_install_md(title, sys_lines, req_lines, backend_port, frontend_port)


def _fallback_install_md(title, sys_lines, req_lines, backend_port, frontend_port) -> str:
    """Deterministic INSTALL.md if the model returns nothing — still accurate and complete."""
    parts = [f"# Installing {title}", ""]
    parts += ["## 1. System prerequisites", ""]
    parts += sys_lines if sys_lines else ["- Python 3.10+ — no other system software is required."]
    parts += [
        "", "## 2. Python & backend dependencies", "",
        "```bash",
        "python -m venv .venv",
        "# macOS/Linux: source .venv/bin/activate   |   Windows: .venv\\Scripts\\activate",
        "pip install -r backend/requirements.txt",
        "```",
        "",
        "This is exactly what the **📦 Install backend deps** quick action (Run tab) runs. "
        "It installs:", "",
    ]
    parts += [f"- `{l}`" for l in req_lines] if req_lines else ["- (no backend/requirements.txt found)"]
    parts += [
        "", "## 3. Configuration", "",
        "```bash", "cp .env.example .env   # then fill in any keys", "```",
        "", "## 4. Run it", "",
        "```bash",
        f"# Backend (from backend/):  uvicorn main:app --host 127.0.0.1 --port {backend_port}",
        f"# Frontend (from frontend/): python -m http.server {frontend_port}",
        "# Or one command from the repo root:  ./run.sh   (Windows: ./run.ps1)",
        "```",
        "", "## 5. Verify it works", "",
        f"- Open http://127.0.0.1:{frontend_port} in your browser.",
        "- Run the tests: `pytest tests -v`.", "",
    ]
    return "\n".join(parts)


# ── Spec formatting / parsing helpers ───────────────────────────────────────
def _format_spec(spec: dict, selected: dict) -> str:
    """Render the spec as readable markdown — the contract handed to every agent
    and shown in the UI's Architecture tab."""
    if spec.get("_raw"):
        return f"# Build Spec: {selected.get('title')}\n\n{spec['_raw']}"

    lines = [f"# Build Spec: {selected.get('title')}", ""]
    if spec.get("summary"):
        lines += [spec["summary"], ""]

    ts = spec.get("tech_stack", {})
    if ts:
        lines.append("## Tech stack")
        if ts.get("frontend"):
            lines.append(f"- **Frontend:** {', '.join(ts['frontend'])}")
        if ts.get("backend"):
            lines.append(f"- **Backend:** {', '.join(ts['backend'])}")
        lines.append("")

    sysreqs = spec.get("system_requirements") or []
    if sysreqs:
        lines.append("## System prerequisites (install by hand before running)")
        for r in sysreqs:
            if not isinstance(r, dict):
                continue
            ver = f" {r.get('version')}" if r.get("version") else ""
            why = f" — {r.get('why')}" if r.get("why") else ""
            how = f"  \n  Install: {r.get('install')}" if r.get("install") else ""
            lines.append(f"- **{r.get('name', '?')}**{ver}{why}{how}")
        lines.append("")

    ports = spec.get("ports", {})
    lines += ["## Ports", f"- Backend: {ports.get('backend', DEFAULT_BACKEND_PORT)}", f"- Frontend: {ports.get('frontend', DEFAULT_FRONTEND_PORT)}", ""]

    if spec.get("env_vars"):
        lines.append("## Environment variables")
        for v in spec["env_vars"]:
            req = " (required)" if v.get("required") else ""
            lines.append(f"- `{v.get('name')}`{req} — {v.get('description', '')}")
        lines.append("")

    if spec.get("data_models"):
        lines.append("## Data models")
        for m in spec["data_models"]:
            fields = ", ".join(f"{k}: {v}" for k, v in (m.get("fields") or {}).items())
            lines.append(f"- **{m.get('name')}**: {fields}")
        lines.append("")

    if spec.get("api_endpoints"):
        lines.append("## API contract (implement these EXACTLY)")
        for e in spec["api_endpoints"]:
            lines.append(f"### {e.get('method', 'GET')} {e.get('path', '/')}")
            if e.get("description"):
                lines.append(e["description"])
            if e.get("request"):
                lines.append(f"- Request: `{json.dumps(e['request'])}`")
            if e.get("response"):
                lines.append(f"- Response: `{json.dumps(e['response'])}`")
            lines.append("")

    fm = spec.get("file_manifest", {})
    if fm:
        lines.append("## File manifest (create ALL of these)")
        for area in ("frontend", "backend"):
            if fm.get(area):
                lines.append(f"**{area}/**")
                for f in fm[area]:
                    lines.append(f"- `{f.get('path')}` — {f.get('purpose', '')}")
        lines.append("")

    for key, heading in (("integration_notes", "Integration notes"), ("frontend_notes", "Frontend notes"), ("backend_notes", "Backend notes")):
        if spec.get(key):
            lines += [f"## {heading}", spec[key], ""]

    return "\n".join(lines)


def _parse_json_array(text: str) -> list:
    text = text.strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        start, end = text.find("["), text.rfind("]") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                return []
    return []


def _parse_json_object(text: str) -> dict:
    text = text.strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                return {}
    return {}


def _port(spec: dict, which: str, default: int) -> int:
    try:
        return int((spec.get("ports") or {}).get(which, default))
    except (TypeError, ValueError):
        return default


# ── Static glue file templates ──────────────────────────────────────────────
_GITIGNORE = """# Python
__pycache__/
*.py[cod]
.venv/
venv/
.pytest_cache/
.mypy_cache/
.ruff_cache/

# Env / secrets
.env

# Node
node_modules/
dist/
build/

# Editor / OS
.vscode/
.idea/
.DS_Store
"""


def _run_sh(backend_port: int, frontend_port: int) -> str:
    return f"""#!/usr/bin/env bash
# One-command setup + run for mac/linux. Run from the project root: ./run.sh
set -e
cd "$(dirname "$0")"

echo "==> Creating virtual environment (.venv)"
python3 -m venv .venv
source .venv/bin/activate

echo "==> Installing backend dependencies"
pip install --upgrade pip
pip install -r backend/requirements.txt

if [ ! -f .env ]; then
  echo "==> Creating .env from .env.example (fill in your keys!)"
  cp .env.example .env
fi

echo "==> Starting backend on http://127.0.0.1:{backend_port}"
( cd backend && uvicorn main:app --host 127.0.0.1 --port {backend_port} ) &
BACKEND_PID=$!

echo "==> Serving frontend on http://127.0.0.1:{frontend_port}"
( cd frontend && python3 -m http.server {frontend_port} ) &
FRONTEND_PID=$!

echo ""
echo "Backend:  http://127.0.0.1:{backend_port}"
echo "Frontend: http://127.0.0.1:{frontend_port}"
echo "Press Ctrl+C to stop both."
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT
wait
"""


def _run_ps1(backend_port: int, frontend_port: int) -> str:
    return f"""# One-command setup + run for Windows PowerShell. Run from the project root: ./run.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "==> Creating virtual environment (.venv)"
python -m venv .venv
& .\\.venv\\Scripts\\Activate.ps1

Write-Host "==> Installing backend dependencies"
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt

if (-not (Test-Path .env)) {{
  Write-Host "==> Creating .env from .env.example (fill in your keys!)"
  Copy-Item .env.example .env
}}

Write-Host "==> Starting backend on http://127.0.0.1:{backend_port}"
Start-Process -NoNewWindow python -ArgumentList "-m","uvicorn","main:app","--host","127.0.0.1","--port","{backend_port}" -WorkingDirectory "backend"

Write-Host "==> Serving frontend on http://127.0.0.1:{frontend_port}"
Set-Location frontend
python -m http.server {frontend_port}
"""


# ── Event helpers ────────────────────────────────────────────────────────────
async def _safe_stage(gen, agent_name: str):
    """Wrap an agent stream — any crash becomes a visible error instead of killing the pipeline."""
    try:
        async for ev in gen:
            yield ev
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        yield _status(
            agent_name,
            f"⚠️ {agent_name} hit an error and was skipped: {type(e).__name__}: {e}",
            data={"error": str(e)},
        )


def _status(agent: str, message: str, data: dict = None) -> dict:
    return {"type": "status", "agent": agent, "message": message, "data": data or {}}

def _thought(agent: str, chunk: str) -> dict:
    return {"type": "thought", "agent": agent, "chunk": chunk}

def _data(agent: str, payload: dict) -> dict:
    return {"type": "data", "agent": agent, "data": payload}

def _chat(agent: str, message: str) -> dict:
    """A message from an agent to the user, rendered in the Chat tab."""
    return {"type": "chat", "agent": agent, "role": "assistant", "message": message}
