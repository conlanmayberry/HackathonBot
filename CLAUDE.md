# HackathonBot

Multi-agent AI system for building hackathon projects. Uses `claude-sonnet-4-6` via the Anthropic Python SDK.

## Architecture

Orchestrator pattern: **Supervisor** delegates to specialist subagents in sequence.

```
Supervisor
├── Planner        → generates 3-5 project ideas from Devpost winners
├── Researcher     → finds similar projects on Devpost + GitHub
├── Frontend Dev   → scaffolds frontend files to output/<slug>/frontend/
├── Backend Dev    → scaffolds backend files to output/<slug>/backend/
├── Debugger       → runs pylint/eslint on generated code
└── Tester         → writes and runs pytest tests
```

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
