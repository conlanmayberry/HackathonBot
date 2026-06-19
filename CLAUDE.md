# HackathonBot

Multi-agent AI system for building hackathon projects. Every agent uses `claude-opus-4-8`
via the Anthropic Python SDK, with **extended thinking** enabled so agents reason before
producing code. All streaming goes through `agents/llm.py`.

## Architecture

The **Planner** is the lead agent. It absorbs what used to be the Supervisor (orchestration)
and Researcher (Devpost analysis), and additionally acts as the project **architect** and
**integrator**. It drives the whole pipeline in sequence:

```
Planner
  1. Research past winners on Devpost + generate 3-5 ideas (waits for user selection)
  2. Kickoff chat (interactive runs only): asks the user a few clarifying questions
     in the Chat tab; the answer is folded into the build instructions. Non-blocking
     (proceeds with best judgment on timeout).
  3. Architect a shared BUILD SPEC — API contract, data models, file manifest,
     env vars, ports. Every downstream agent builds against this one contract.
  4. Frontend Dev + Backend Dev run in parallel, both fed the same spec
       → output/<slug>/frontend/  and  output/<slug>/backend/
  5. Integrate — write root glue files so it runs as ONE app:
       README.md, .env.example, .gitignore, run.sh, run.ps1
  6. QA + fix loop — one QA agent reconciles requirements.txt against actual imports,
     writes tests/conftest.py + suite, then runs pylint + pytest + a frontend-asset
     check. On failure it hands the REAL errors back to the relevant dev
     (backend_dev.repair / frontend_dev.repair) and re-verifies. Capped at
     MAX_FIX_ROUNDS (2).
```

Key reliability rules: dev output uses a large `max_tokens` budget (so multi-file output
is never truncated → no missing files); the frontend and backend share one spec (so the
API contract matches); QA reconciles `requirements.txt` against actual imports (so
`pip install` is complete) and loops fixes through the devs until pylint+pytest pass; the
root README + run scripts give one clear install path.

Interactive chat: the planner can message the user mid-build via `ask_user` (wired in
`main.py` → job `chat_event`/`awaiting_chat`); `/api/chat/{job_id}` routes a reply to a
waiting planner, otherwise falls back to the standalone chat assistant (`agents/chat.py`).
Agent→user messages are emitted as `{"type":"chat"}` events the UI renders in the Chat tab.

## Running

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your keys
uvicorn main:app --reload
# open http://localhost:8000
```

## Environment

- `ANTHROPIC_API_KEY` — Anthropic API key
- `GITHUB_TOKEN` — GitHub PAT for higher API rate limits (5000 req/hr vs 60)

## Output

Generated projects land in `./output/<project-slug>/` (gitignored).
