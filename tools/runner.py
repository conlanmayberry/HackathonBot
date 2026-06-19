import sys
import asyncio
import uuid
import threading
import subprocess
from pathlib import Path
from tools.file_writer import get_output_path, OUTPUT_DIR


class RunManager:
    """Starts/streams/stops subprocesses that run generated projects.

    Uses a background thread + classic subprocess (not asyncio subprocess) so it
    works under any event loop — uvicorn on Windows uses a SelectorEventLoop,
    which cannot spawn asyncio subprocesses. All commands are constrained to run
    inside output/<slug>/ for safety.
    """

    def __init__(self):
        self.runs: dict[str, dict] = {}

    def resolve_cwd(self, project_title: str, subdir: str = "") -> Path:
        base = get_output_path(project_title).resolve()
        target = (base / subdir).resolve()
        # Guard: never allow escaping the project's output directory.
        if target != base and base not in target.parents:
            raise ValueError("Refusing to run outside the project's output directory.")
        return target

    async def start(self, project_title: str, command: str, subdir: str = "") -> dict:
        try:
            cwd = self.resolve_cwd(project_title, subdir)
        except ValueError as e:
            return {"ok": False, "error": str(e)}
        if not cwd.exists():
            return {"ok": False, "error": f"Directory does not exist: {cwd}"}

        run_id = str(uuid.uuid4())
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        try:
            proc = subprocess.Popen(
                command,
                shell=True,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # line-buffered
            )
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"Failed to start: {e}"}

        self.runs[run_id] = {"proc": proc, "queue": queue, "done": False, "command": command, "cwd": str(cwd)}

        def push(item):
            # Hand the line back to the event loop thread safely.
            loop.call_soon_threadsafe(queue.put_nowait, item)

        push(f"$ {command}\n  (in {cwd})\n\n")

        def reader():
            try:
                for line in proc.stdout:
                    push(line)
                proc.wait()
                push(f"\n[process exited with code {proc.returncode}]\n")
            except Exception as e:  # noqa: BLE001
                push(f"\n[stream error: {e}]\n")
            finally:
                self.runs[run_id]["done"] = True
                push(None)  # sentinel

        threading.Thread(target=reader, daemon=True).start()
        return {"ok": True, "run_id": run_id, "cwd": str(cwd)}

    async def stream(self, run_id: str):
        """Async generator yielding output lines until the run finishes."""
        run = self.runs.get(run_id)
        if not run:
            yield "[run not found]\n"
            return
        queue = run["queue"]
        while True:
            line = await queue.get()
            if line is None:
                break
            yield line

    def stop(self, run_id: str) -> dict:
        run = self.runs.get(run_id)
        if not run:
            return {"ok": False, "error": "Run not found"}
        proc = run["proc"]
        if proc.poll() is None:
            try:
                proc.terminate()
            except Exception:  # noqa: BLE001
                pass
        return {"ok": True}

    def stop_all(self):
        for run in self.runs.values():
            proc = run["proc"]
            if proc.poll() is None:
                try:
                    proc.terminate()
                except Exception:  # noqa: BLE001
                    pass


def preset_commands() -> dict:
    """One-click commands for common project shapes. Uses the current interpreter."""
    py = Path(sys.executable).name or "python"
    return {
        "install_backend": {"label": "📦 Install backend deps", "subdir": "backend", "command": f"{py} -m pip install -r requirements.txt"},
        "run_backend": {"label": "▶ Start backend (port 8100)", "subdir": "backend", "command": f"{py} -m uvicorn main:app --host 127.0.0.1 --port 8100"},
        "run_tests": {"label": "🧪 Run tests", "subdir": "", "command": f"{py} -m pytest tests -v -p no:cacheprovider"},
        "serve_frontend": {"label": "🌐 Serve frontend (port 8200)", "subdir": "frontend", "command": f"{py} -m http.server 8200"},
    }


run_manager = RunManager()
