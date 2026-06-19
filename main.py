import asyncio
import json
import uuid
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agents.planner import PlannerAgent
from agents.chat import chat_reply
from tools.hackathon_search import search_hackathons
from tools.github_upload import push_to_github
from tools.runner import run_manager, preset_commands

app = FastAPI(title="HackathonBot")
app.mount("/static", StaticFiles(directory="static"), name="static")

# In-memory job store. Post-build actions (push/run) key off the project title
# (which maps to a folder on disk), so they keep working even after a reload.
_jobs: dict[str, dict] = {}


class StartRequest(BaseModel):
    hackathon: str
    university: str
    theme: str
    autonomous: bool = False
    instructions: str = ""


class ChatRequest(BaseModel):
    message: str


class PushRequest(BaseModel):
    project_title: str
    repo_name: str = None
    private: bool = True


class RunRequest(BaseModel):
    project_title: str
    command: str
    subdir: str = ""


@app.get("/", response_class=HTMLResponse)
async def index():
    return Path("static/index.html").read_text(encoding="utf-8")


# ── Job lifecycle ───────────────────────────────────────────────────────────
@app.post("/api/start")
async def start_job(req: StartRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "events": [],
        "done": False,
        "result": None,
        "request": req.model_dump(),
        "chat": [],
        "selection_event": asyncio.Event(),
        "chosen_idea_index": None,
        "custom_idea": None,
        "action": "build",
    }
    background_tasks.add_task(_run_job, job_id, req)
    return {"job_id": job_id}


@app.get("/api/stream/{job_id}")
async def stream_job(job_id: str):
    if job_id not in _jobs:
        return StreamingResponse(iter([]), media_type="text/event-stream")

    async def generator():
        sent = 0
        while True:
            job = _jobs.get(job_id, {})
            events = job.get("events", [])
            while sent < len(events):
                yield f"data: {json.dumps(events[sent])}\n\n"
                sent += 1
            if job.get("done"):
                yield 'data: {"type":"status","agent":"supervisor","message":"__done__","data":{}}\n\n'
                break
            await asyncio.sleep(0.3)

    return StreamingResponse(generator(), media_type="text/event-stream")


@app.get("/api/result/{job_id}")
async def get_result(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        return {"error": "Job not found"}
    return job.get("result") or {"status": "in_progress"}


@app.post("/api/select-idea/{job_id}")
async def select_idea(job_id: str, payload: dict):
    """Unpause the supervisor: build a chosen/custom idea, or regenerate."""
    job = _jobs.get(job_id)
    if not job:
        return {"ok": False, "error": "Job not found"}
    job["action"] = payload.get("action", "build")
    idx = payload.get("index")
    job["chosen_idea_index"] = int(idx) if idx is not None else None
    job["custom_idea"] = payload.get("custom_idea")
    event = job.get("selection_event")
    if event:
        event.set()
    return {"ok": True}


# ── Research / chat ─────────────────────────────────────────────────────────
@app.get("/api/hackathons")
async def find_hackathons(query: str = "", status: str = None):
    results = await search_hackathons(query=query, status=status)
    return {"hackathons": results}


@app.post("/api/chat/{job_id}")
async def chat(job_id: str, req: ChatRequest):
    job = _jobs.get(job_id)
    if not job:
        return {"error": "Job not found. Launch a project first."}
    history = job.setdefault("chat", [])
    reply = await chat_reply(job, history, req.message)
    history.append({"role": "user", "content": req.message})
    history.append({"role": "assistant", "content": reply})
    return {"reply": reply}


# ── Post-build actions (keyed by project title, not job) ────────────────────
@app.post("/api/push-github")
async def push_github(req: PushRequest):
    if not req.project_title:
        return {"ok": False, "error": "No project specified."}
    # Blocking PyGithub calls run off the event loop. Reads files from disk,
    # so it works even if the original job is gone from memory.
    return await asyncio.to_thread(push_to_github, req.project_title, req.repo_name, req.private)


@app.get("/api/run-presets")
async def run_presets():
    return {"presets": preset_commands()}


@app.post("/api/run")
async def run_start(req: RunRequest):
    if not req.project_title or not req.command:
        return {"ok": False, "error": "project_title and command are required."}
    return await run_manager.start(req.project_title, req.command, req.subdir)


@app.get("/api/run-stream/{run_id}")
async def run_stream(run_id: str):
    async def generator():
        async for line in run_manager.stream(run_id):
            yield f"data: {json.dumps({'text': line})}\n\n"
        yield 'data: {"done": true}\n\n'

    return StreamingResponse(generator(), media_type="text/event-stream")


@app.post("/api/run-stop/{run_id}")
async def run_stop(run_id: str):
    return run_manager.stop(run_id)


@app.on_event("shutdown")
def _cleanup():
    run_manager.stop_all()


# ── Background worker ───────────────────────────────────────────────────────
async def _run_job(job_id: str, req: StartRequest):
    supervisor = SupervisorAgent()
    job = _jobs[job_id]
    events = job["events"]
    last_result = {}

    async def wait_for_selection() -> dict:
        ev = job["selection_event"]
        await ev.wait()
        ev.clear()  # re-arm so a "regenerate" can wait again next round
        return {
            "action": job.get("action", "build"),
            "index": job.get("chosen_idea_index"),
            "custom_idea": job.get("custom_idea"),
        }

    try:
        async for event in supervisor.run(
            hackathon=req.hackathon,
            university=req.university,
            theme=req.theme,
            autonomous=req.autonomous,
            instructions=req.instructions,
            select_idea=wait_for_selection,
        ):
            events.append(event)
            if event.get("data", {}).get("result"):
                last_result = event["data"]["result"]
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        events.append({
            "type": "status",
            "agent": "supervisor",
            "message": f"❌ The run stopped due to an error: {type(e).__name__}: {e}",
            "data": {"error": str(e)},
        })
    finally:
        job["done"] = True
        job["result"] = last_result
