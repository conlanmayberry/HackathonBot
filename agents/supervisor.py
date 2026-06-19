import asyncio
from typing import AsyncGenerator
from agents.planner import PlannerAgent
from agents.researcher import ResearcherAgent
from agents.frontend_dev import FrontendDevAgent
from agents.backend_dev import BackendDevAgent
from agents.debugger import DebuggerAgent
from agents.tester import TesterAgent
from tools.file_writer import list_output_files, get_output_path


class SupervisorAgent:
    def __init__(self):
        self.planner = PlannerAgent()
        self.researcher = ResearcherAgent()
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

        yield _status("supervisor", f"Starting HackathonBot for {hackathon} at {university}")
        if instructions:
            yield _status("supervisor", f"Applying your additional instructions: \"{instructions[:120]}\"")

        # ── Steps 1 & 2: Plan + select (loops on "regenerate") ─────────────
        fallback = {"title": theme, "description": theme, "tech_stack": ["Python", "JavaScript"]}
        selected = None
        seen_titles: list[str] = []

        while selected is None:
            ideas = []
            winners_count = 0
            avoid = seen_titles or None
            async for event in _safe_stage(self.planner.run(hackathon, university, theme, instructions, avoid), "planner"):
                yield event
                if event["type"] == "data":
                    ideas = event["data"].get("ideas", [])
                    winners_count = event["data"].get("winners_researched", 0)

            seen_titles.extend(i.get("title", "") for i in ideas)
            yield _status("planner", f"Generated {len(ideas)} ideas ({winners_count} past winners analyzed)", data={"ideas": ideas})

            if not ideas:
                selected = fallback
                yield _status("supervisor", "No ideas were generated; falling back to the raw theme.")
                break

            if autonomous or select_idea is None:
                selected = ideas[0]
                yield _status("supervisor", f"Auto-selected: '{selected['title']}'", data={"selection_made": True})
                break

            # Pause until the user clicks "Build this idea" or "Regenerate".
            yield _status("supervisor", "Waiting for you to choose an idea, add your own, or regenerate…", data={"awaiting_selection": True})
            choice = await select_idea() or {}
            action = choice.get("action", "build")

            if action == "regenerate":
                yield _status("planner", "Got it — regenerating a fresh set of ideas…", data={"regenerating": True})
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
                yield _status("supervisor", f"Building your own idea: '{selected['title']}'.", data={"selection_made": True})
            else:
                idx = choice.get("index")
                idx = idx if isinstance(idx, int) and 0 <= idx < len(ideas) else 0
                selected = ideas[idx]
                yield _status("supervisor", f"You selected: '{selected['title']}' — building it now.", data={"selection_made": True})

        # ── Step 3: Researcher ─────────────────────────────────────────────
        research_report = ""
        devpost_count = github_count = 0
        async for event in _safe_stage(self.researcher.run(
            hackathon, selected["title"], selected.get("description", ""), university, theme
        ), "researcher"):
            yield event
            if event["type"] == "data":
                research_report = event["data"].get("report", "")
                devpost_count = event["data"].get("devpost_count", 0)
                github_count = event["data"].get("github_count", 0)

        yield _status(
            "researcher",
            f"Research complete — {devpost_count} Devpost + {github_count} GitHub results",
            data={"report": research_report},
        )

        # ── Step 4: Frontend + Backend in parallel (queue-merged streams) ──
        yield _status("supervisor", "Launching Frontend Dev and Backend Dev in parallel…")

        tech_stack = selected.get("tech_stack", ["Python", "JavaScript", "FastAPI"])
        q: asyncio.Queue = asyncio.Queue()
        fe_result: dict = {}
        be_result: dict = {}

        async def drain(gen, sentinel):
            async for ev in gen:
                await q.put(ev)
            await q.put(sentinel)

        asyncio.create_task(drain(
            _safe_stage(self.frontend_dev.run(selected["title"], selected.get("description", ""), tech_stack, research_report, instructions), "frontend_dev"),
            {"type": "__done__", "agent": "frontend_dev"},
        ))
        asyncio.create_task(drain(
            _safe_stage(self.backend_dev.run(selected["title"], selected.get("description", ""), tech_stack, research_report, instructions), "backend_dev"),
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

        # ── Step 5: Debugger ───────────────────────────────────────────────
        debug_data: dict = {}
        async for event in _safe_stage(self.debugger.run(selected["title"]), "debugger"):
            yield event
            if event["type"] == "data":
                debug_data = event["data"]

        yield _status("debugger", f"Debug review complete ({debug_data.get('files_checked', 0)} files checked)", data={"issues": debug_data.get("issues", "")})

        # ── Step 6: Tester ─────────────────────────────────────────────────
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
            "research": {"report": research_report, "devpost_count": devpost_count, "github_count": github_count},
            "frontend": fe_result,
            "backend": be_result,
            "debug": debug_data,
            "tests": test_data,
            "all_files": all_files,
            "output_path": output_path,
        }

        yield _status(
            "supervisor",
            f"Done! '{selected['title']}' scaffolded at {output_path} ({len(all_files)} files)",
            data={"result": final_result},
        )


async def _safe_stage(gen, agent_name: str):
    """Consume an agent's event stream, converting any crash into a visible
    error status instead of letting it kill the whole pipeline."""
    try:
        async for ev in gen:
            yield ev
    except Exception as e:  # noqa: BLE001 — we want to catch everything here
        import traceback
        traceback.print_exc()
        yield _status(
            agent_name,
            f"⚠️ {agent_name} hit an error and was skipped: {type(e).__name__}: {e}",
            data={"error": str(e)},
        )


def _status(agent: str, message: str, data: dict = None) -> dict:
    return {"type": "status", "agent": agent, "message": message, "data": data or {}}
