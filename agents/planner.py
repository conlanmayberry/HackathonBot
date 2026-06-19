import os
import json
import asyncio
from typing import AsyncGenerator
import anthropic
from tools.devpost import search_devpost_winners
from tools.file_writer import list_output_files, get_output_path
from agents.frontend_dev import FrontendDevAgent
from agents.backend_dev import BackendDevAgent
from agents.debugger import DebuggerAgent
from agents.tester import TesterAgent

MODEL = "claude-opus-4-8"


class PlannerAgent:
    def __init__(self):
        self.frontend_dev = FrontendDevAgent()
        self.backend_dev = BackendDevAgent()
        self.debugger = DebuggerAgent()
        self.tester = TesterAgent()

    async def run(
        self,
        hackathon: str,
        university: str,
        theme: str,
        autonomous: bool = False,
        instructions: str = "",
        select_idea=None,
    ) -> AsyncGenerator[dict, None]:

        yield _status("planner", f"Starting HackathonBot for {hackathon} at {university}")
        if instructions:
            yield _status("planner", f"Applying additional instructions: \"{instructions[:120]}\"")

        # ── Step 1: Search Devpost for past winners ─────────────────────────
        yield _status("planner", f"Searching Devpost for past '{theme}' winners…")
        winners = await search_devpost_winners(theme, max_results=15)
        winners_text = "\n".join(
            f"- {p['title']}: {p['description'][:200]}" for p in winners
        ) or "No results found — use general hackathon knowledge."
        yield _status("planner", f"Found {len(winners)} past winners. Generating ideas…")

        # ── Step 2: Generate ideas + wait for selection (loops on regenerate) ─
        fallback = {"title": theme, "description": theme, "tech_stack": ["Python", "JavaScript"]}
        selected = None
        seen_titles: list[str] = []

        while selected is None:
            ideas = []
            avoid = seen_titles or None
            regenerating = bool(avoid)

            extra = (
                f"\n\nIMPORTANT — additional instructions from the team (follow these closely):\n{instructions}"
                if instructions else ""
            )
            avoid_block = ""
            if avoid:
                joined = "; ".join(avoid)
                avoid_block = (
                    f"\n\nThe team has already seen and rejected these ideas — do NOT repeat or lightly "
                    f"reword them. Produce clearly DIFFERENT concepts:\n{joined}"
                )

            prompt = f"""You are a hackathon strategist helping a college team win {hackathon} at {university}.

Theme/category: {theme}

Past winning projects found on Devpost:
{winners_text}{extra}{avoid_block}

Generate 3-5 project ideas. About 60% should be inspired by patterns you see in past winners, 40% should be original/creative ideas that feel fresh.

For each idea return a JSON object with:
- title: short project name
- description: 2-3 sentence pitch
- tech_stack: list of recommended technologies
- why_it_wins: 1-2 sentences on why judges would pick this
- originality: "inspired" or "original"

Return a JSON array of these objects. No markdown fences, just raw JSON."""

            client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), timeout=120.0, max_retries=2)
            full_text = ""
            async with client.messages.stream(
                model=MODEL,
                max_tokens=2000,
                temperature=1.0 if regenerating else 0.7,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                async for chunk in stream.text_stream:
                    full_text += chunk
                    yield _thought("planner", chunk)

            try:
                ideas = json.loads(full_text.strip())
            except json.JSONDecodeError:
                start = full_text.find("[")
                end = full_text.rfind("]") + 1
                ideas = json.loads(full_text[start:end]) if start != -1 else []

            seen_titles.extend(i.get("title", "") for i in ideas)
            yield _status("planner", f"Generated {len(ideas)} ideas ({len(winners)} past winners analyzed)", data={"ideas": ideas})
            yield _data("planner", {"ideas": ideas, "winners_researched": len(winners)})

            if not ideas:
                selected = fallback
                yield _status("planner", "No ideas generated; falling back to the raw theme.")
                break

            if autonomous or select_idea is None:
                selected = ideas[0]
                yield _status("planner", f"Auto-selected: '{selected['title']}'", data={"selection_made": True})
                break

            yield _status("planner", "Waiting for you to choose an idea, add your own, or regenerate…", data={"awaiting_selection": True})
            choice = await select_idea() or {}
            action = choice.get("action", "build")

            if action == "regenerate":
                yield _status("planner", "Regenerating a fresh set of ideas…", data={"regenerating": True})
                continue

            custom = choice.get("custom_idea")
            if custom and custom.get("title"):
                selected = {
                    "title": custom.get("title"),
                    "description": custom.get("description", ""),
                    "tech_stack": custom.get("tech_stack") or ["Python", "JavaScript", "FastAPI"],
                    "why_it_wins": custom.get("why_it_wins", ""),
                    "originality": "original",
                }
                yield _status("planner", f"Building your own idea: '{selected['title']}'.", data={"selection_made": True})
            else:
                idx = choice.get("index")
                idx = idx if isinstance(idx, int) and 0 <= idx < len(ideas) else 0
                selected = ideas[idx]
                yield _status("planner", f"You selected: '{selected['title']}' — building it now.", data={"selection_made": True})

        # ── Step 3: Frontend + Backend in parallel ─────────────────────────
        yield _status("planner", "Launching Frontend Dev and Backend Dev in parallel…")

        tech_stack = selected.get("tech_stack", ["Python", "JavaScript", "FastAPI"])
        q: asyncio.Queue = asyncio.Queue()
        fe_result: dict = {}
        be_result: dict = {}

        async def drain(gen, sentinel):
            async for ev in gen:
                await q.put(ev)
            await q.put(sentinel)

        asyncio.create_task(drain(
            _safe_stage(
                self.frontend_dev.run(selected["title"], selected.get("description", ""), tech_stack, "", instructions),
                "frontend_dev",
            ),
            {"type": "__done__", "agent": "frontend_dev"},
        ))
        asyncio.create_task(drain(
            _safe_stage(
                self.backend_dev.run(selected["title"], selected.get("description", ""), tech_stack, "", instructions),
                "backend_dev",
            ),
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

        # ── Step 4: Debugger ───────────────────────────────────────────────
        debug_data: dict = {}
        async for event in _safe_stage(self.debugger.run(selected["title"]), "debugger"):
            yield event
            if event["type"] == "data":
                debug_data = event["data"]

        yield _status("debugger", f"Debug review complete ({debug_data.get('files_checked', 0)} files checked)", data={"issues": debug_data.get("issues", "")})

        # ── Step 5: Tester ─────────────────────────────────────────────────
        test_data: dict = {}
        async for event in _safe_stage(self.tester.run(selected["title"], selected.get("description", "")), "tester"):
            yield event
            if event["type"] == "data":
                test_data = event["data"]

        yield _status("tester", "Tests written and executed", data={"pytest_output": test_data.get("pytest_output", "")})

        # ── Final summary ──────────────────────────────────────────────────
        all_files = list_output_files(selected["title"])
        output_path = str(get_output_path(selected["title"]))

        final_result = {
            "selected_idea": selected,
            "frontend": fe_result,
            "backend": be_result,
            "debug": debug_data,
            "tests": test_data,
            "all_files": all_files,
            "output_path": output_path,
        }

        yield _status(
            "planner",
            f"Done! '{selected['title']}' scaffolded at {output_path} ({len(all_files)} files)",
            data={"result": final_result},
        )


async def _safe_stage(gen, agent_name: str):
    """Wrap an agent stream — any crash becomes a visible error instead of killing the pipeline."""
    try:
        async for ev in gen:
            yield ev
    except Exception as e:
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
