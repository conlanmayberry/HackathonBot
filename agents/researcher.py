import os
import asyncio
from typing import AsyncGenerator
import anthropic
from tools.devpost import search_devpost
from tools.github_search import search_github_hackathon
from tools.hackathon_search import lookup_hackathon

MODEL = "claude-sonnet-4-6"


class ResearcherAgent:
    async def run(self, hackathon: str, project_title: str, project_description: str, university: str, theme: str) -> AsyncGenerator[dict, None]:
        yield _status("researcher", f"Looking up '{hackathon}' on Devpost (past/present/future)…")

        # Run all three lookups in parallel, each bounded by a timeout and isolated so
        # one failing/slow source can never stall the whole step.
        async def _safe(coro, default, timeout=20):
            try:
                return await asyncio.wait_for(coro, timeout=timeout)
            except Exception:
                return default

        event_lookup, devpost_results, github_results = await asyncio.gather(
            _safe(lookup_hackathon(hackathon), {"found": False, "match": None, "candidates": []}),
            _safe(search_devpost(f"{theme} {project_title}", max_results=12), []),
            _safe(asyncio.to_thread(search_github_hackathon, f"{theme} {project_title}", max_results=10), []),
        )

        # Defensive defaults in case a source returned something unexpected.
        event_lookup = event_lookup or {"found": False, "match": None, "candidates": []}
        devpost_results = devpost_results or []
        github_results = github_results or []

        # Build a factual note about the event so the model doesn't hallucinate.
        if event_lookup["found"]:
            m = event_lookup["match"]
            event_note = (
                f"CONFIRMED: '{m['title']}' exists on Devpost — {m['open_state']}, "
                f"dates: {m['deadline']}, location: {m['location']}, organizer: {m['organization']}, "
                f"prizes: {m['prize_amount']}. URL: {m['url']}"
            )
            yield _status("researcher", f"Found event: {m['title']} ({m['open_state']})")
        else:
            cand = event_lookup.get("candidates", [])
            if cand:
                cand_list = "; ".join(f"{c['title']} ({c['open_state']})" for c in cand[:5])
                event_note = (
                    f"NOT CONFIRMED: Could not find an exact Devpost match for '{hackathon}'. "
                    f"Closest results: {cand_list}. Tell the user you could not verify the specific event "
                    f"and that the strategy below is based on the theme and these similar events."
                )
                yield _status("researcher", f"No exact match for '{hackathon}' — found {len(cand)} similar events")
            else:
                event_note = (
                    f"NOT CONFIRMED: No Devpost results at all for '{hackathon}'. "
                    f"Tell the user clearly that you could not find this specific hackathon and that "
                    f"the strategy is based purely on the theme '{theme}' and general hackathon knowledge."
                )
                yield _status("researcher", f"Could not find '{hackathon}' anywhere on Devpost")

        yield _status("researcher", f"Analyzing {len(devpost_results)} Devpost + {len(github_results)} GitHub results…")

        devpost_text = "\n".join(
            f"- [{p['university'] or 'Unknown school'}] {p['title']}: {p['description'][:200]}"
            for p in devpost_results
        ) or "No Devpost project results found."

        github_text = "\n".join(
            f"- {r['name']} ({r['stars']} stars): {r['description'][:200]}"
            for r in github_results
        ) or "No GitHub results found."

        prompt = f"""You are a research analyst for a college hackathon team at {university}.

Hackathon: {hackathon}
Event verification (from Devpost):
{event_note}

Our project idea:
Title: {project_title}
Description: {project_description}

Similar projects found on Devpost (including nearby universities):
{devpost_text}

Related GitHub repositories:
{github_text}

Write a research report with these sections:
1. **Event Status** — In 1-2 sentences, state clearly whether the specific hackathon "{hackathon}" was verified on Devpost. If it was NOT confirmed, say so explicitly and explain what the rest of the report is based on instead.
2. **What's Already Been Done** — summarize existing approaches (2-3 sentences)
3. **Gaps & Weaknesses** — what problems do existing projects fail to solve (2-3 sentences)
4. **Our Differentiators** — 3-5 bullet points on what will make our project stand out
5. **Technical Insights** — patterns or tech choices from successful projects worth borrowing

Be honest about uncertainty. Use markdown formatting."""

        client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), timeout=120.0, max_retries=2)
        full_text = ""
        async with client.messages.stream(
            model=MODEL,
            max_tokens=1600,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for chunk in stream.text_stream:
                full_text += chunk
                yield _thought("researcher", chunk)

        yield _data("researcher", {
            "report": full_text,
            "devpost_count": len(devpost_results),
            "github_count": len(github_results),
            "event_found": event_lookup["found"],
            "event_match": event_lookup.get("match"),
        })


def _status(agent: str, message: str) -> dict:
    return {"type": "status", "agent": agent, "message": message}

def _thought(agent: str, chunk: str) -> dict:
    return {"type": "thought", "agent": agent, "chunk": chunk}

def _data(agent: str, payload: dict) -> dict:
    return {"type": "data", "agent": agent, "data": payload}
