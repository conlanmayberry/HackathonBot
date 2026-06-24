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
from tools.file_writer import save_job_meta, load_job_meta, latest_project, list_output_files, get_output_path

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
    project_title: str = ""


class PushRequest(BaseModel):
    project_title: str
    repo_name: str = None
    private: bool = True


class RunRequest(BaseModel):
    project_title: str
    command: str
    subdir: str = ""


class SaveFileRequest(BaseModel):
    project_title: str
    relative_path: str
    content: str


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
        # In-build chat: lets the planner ask the user questions mid-run.
        "chat_event": asyncio.Event(),
        "awaiting_chat": False,
        "chat_answer": None,
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
                yield 'data: {"type":"status","agent":"planner","message":"__done__","data":{}}\n\n'
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
    """Unpause the planner: build a chosen/custom idea, or regenerate."""
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
        # The in-memory job is gone (server reloaded/restarted). Reconstruct it from
        # disk so the user can keep talking to the team and edit the finished project.
        job = _reconstruct_job(req.project_title)
        if not job:
            return {"reply": "I couldn't find a generated project to work on yet. "
                             "Launch a build first, then I can chat about it and edit the files."}
        _jobs[job_id] = job  # cache it so this chat session keeps its history
    history = job.setdefault("chat", [])

    # If the planner is mid-build waiting on the user, deliver this message to it
    # instead of generating a standalone assistant reply.
    if job.get("awaiting_chat"):
        history.append({"role": "user", "content": req.message})
        job["chat_answer"] = req.message
        ev = job.get("chat_event")
        if ev:
            ev.set()
        return {"reply": "Got it — I'll factor that into the build. 👍"}

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


@app.get("/api/read-file")
async def read_file_endpoint(project_title: str, path: str):
    if not project_title or not path:
        return {"ok": False, "error": "Missing parameters."}
    try:
        base = get_output_path(project_title).resolve()
        target = (base / path).resolve()
        if not str(target).startswith(str(base)):
            return {"ok": False, "error": "Invalid path."}
        if not target.exists():
            return {"ok": False, "error": "File not found."}
        return {"ok": True, "content": target.read_text(encoding="utf-8", errors="replace")}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/save-file")
async def save_file_endpoint(req: SaveFileRequest):
    if not req.project_title or not req.relative_path:
        return {"ok": False, "error": "Missing parameters."}
    try:
        base = get_output_path(req.project_title).resolve()
        target = (base / req.relative_path).resolve()
        if not str(target).startswith(str(base)):
            return {"ok": False, "error": "Invalid path."}
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(req.content, encoding="utf-8")
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.on_event("shutdown")
def _cleanup():
    run_manager.stop_all()


def _reconstruct_job(project_title: str = "") -> dict | None:
    """Rebuild a minimal job from disk after the in-memory store was cleared (server
    reload/restart). Tries the title the frontend sent, then the most recent build.
    Returns a job dict shaped like a live one (request + result + chat), or None if
    there's genuinely nothing on disk to talk about."""
    meta = load_job_meta(project_title) if project_title else None
    if meta is None:
        slug = latest_project()
        if slug is None:
            return None
        meta = load_job_meta(slug)
        if meta is None:
            # Folder exists but no saved metadata — synthesize the minimum chat needs.
            files = list_output_files(slug)
            if not files:
                return None
            meta = {"request": {}, "result": {"selected_idea": {"title": slug}, "all_files": files}}

    return {
        "request": meta.get("request", {}),
        "result": meta.get("result", {}),
        "chat": [],
        "awaiting_chat": False,  # the live planner is gone; route to the editing assistant
    }


# ── Background worker ───────────────────────────────────────────────────────
async def _run_job(job_id: str, req: StartRequest):
    planner = PlannerAgent()
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

    async def ask_user(question: str, timeout: float = 240.0):
        """Post a question from the planner into the Chat tab and wait (bounded) for
        a reply. Returns the user's text, or None if they don't answer in time."""
        job["chat"].append({"role": "assistant", "content": question})
        events.append({"type": "chat", "agent": "planner", "role": "assistant", "message": question})
        job["chat_answer"] = None
        job["chat_event"].clear()
        job["awaiting_chat"] = True
        try:
            await asyncio.wait_for(job["chat_event"].wait(), timeout=timeout)
            return job.get("chat_answer")
        except asyncio.TimeoutError:
            return None
        finally:
            job["awaiting_chat"] = False

    try:
        async for event in planner.run(
            hackathon=req.hackathon,
            university=req.university,
            theme=req.theme,
            autonomous=req.autonomous,
            instructions=req.instructions,
            select_idea=wait_for_selection,
            ask_user=ask_user,
        ):
            events.append(event)
            if event.get("data", {}).get("result"):
                last_result = event["data"]["result"]
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        events.append({
            "type": "status",
            "agent": "planner",
            "message": f"❌ The run stopped due to an error: {type(e).__name__}: {e}",
            "data": {"error": str(e)},
        })
    finally:
        job["done"] = True
        job["result"] = last_result
        # Persist so chat survives a server reload and can edit the finished project.
        title = (last_result.get("selected_idea") or {}).get("title")
        if title:
            save_job_meta(title, {"request": job.get("request", {}), "result": last_result})
