# HackathonBot

Multi-agent AI system for building hackathon projects, via the Anthropic Python SDK with
**adaptive thinking** enabled so agents reason before producing code. All streaming goes
through `agents/llm.py`, which defines two model tiers (picked per task to balance quality
against cost):
- `MODEL` = `claude-opus-4-8` — the architect's brain. Used only for the high-stakes
  reasoning where a wrong call cascades through the whole build: the Planner's Devpost
  research / idea generation and the build-spec architecture step.
- `MODEL_CODE` = `claude-sonnet-4-6` — the builders' hands (~40% cheaper, identical
  thinking API). Used for the well-scoped work that builds against a clear spec: frontend
  & backend devs, QA test generation, the planner's clarifying questions + README glue,
  and the interactive chat editor.

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
     writes tests/conftest.py + suite, then runs pylint + pytest + INTEGRATION checks.
     On failure it hands the REAL errors back to the relevant dev
     (backend_dev.repair / frontend_dev.repair) and re-verifies. Capped at
     MAX_FIX_ROUNDS (2).
  7. INSTALL.md — written LAST (so requirements.txt is final after QA's reconciliation).
     The architect declares non-pip prerequisites in the build spec's `system_requirements`
     (game engines, language runtimes, DB servers, native CLI tools); the planner expands
     them into explicit per-OS install steps AND enumerates every backend package the
     "📦 Install backend deps" quick action installs. Rendered inline at the top of the UI's
     Files tab (planner._write_install_guide → result.install_guide).
```

Integration verification (the failure class unit tests never catch — one agent
referencing an artifact another agent never produced). QA.verify runs three deterministic
graph checks beyond pylint/pytest, in `agents/qa.py`:
- `_check_frontend_assets` — every css/js/asset referenced in HTML resolves to a file.
- `_check_js_imports` — every relative ES-module import resolves (a failed top-level
  import silently bricks the whole page).
- `_check_api_contract` — every static frontend fetch()/axios path matches a backend
  route (conservative: external URLs and dynamic-first-segment paths are skipped so there
  are no false-positive fix-loop thrash). Gaps route to backend_dev.repair; broken
  imports/assets route to frontend_dev.repair.
The build spec's `file_manifest` is the ownership contract (every referenced file/endpoint
must have an owner); dev prompts require each agent to "close its dependency graph" and
"fail loud" before finishing.

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
