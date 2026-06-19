import os
import asyncio
import subprocess
from typing import AsyncGenerator
import anthropic
from tools.file_writer import get_output_path, list_output_files

MODEL = "claude-opus-4-8"


class DebuggerAgent:
    async def run(self, project_title: str) -> AsyncGenerator[dict, None]:
        yield _status("debugger", "Running pylint on generated backend…")

        output_path = get_output_path(project_title)
        files = list_output_files(project_title)
        pylint_output = await _run_pylint(output_path)

        py_files = [f for f in files if f.endswith(".py")]
        js_files = [f for f in files if f.endswith(".js")]

        code_snippets = []
        for rel in (py_files + js_files)[:6]:
            full = output_path / rel
            try:
                content = full.read_text(encoding="utf-8")[:1500]
                code_snippets.append(f"### {rel}\n```\n{content}\n```")
            except OSError:
                pass

        code_context = "\n\n".join(code_snippets) or "No code files found."
        yield _status("debugger", f"pylint done. Reviewing {len(py_files) + len(js_files)} files with AI…")

        prompt = f"""You are a senior code reviewer auditing a hackathon project.

pylint output:
{pylint_output[:2000]}

Code files:
{code_context}

Identify real bugs and issues. Ignore style nitpicks — focus on:
1. Logic errors that would break functionality during the demo
2. Missing error handling that would crash the app
3. Security issues (hardcoded secrets, injection risks, etc.)
4. API routes that won't work as written

For each issue: filename, short description, suggested fix. Use markdown bullet list.
If no real bugs found, say so clearly."""

        client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), timeout=120.0, max_retries=2)
        full_text = ""
        async with client.messages.stream(
            model=MODEL,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for chunk in stream.text_stream:
                full_text += chunk
                yield _thought("debugger", chunk)

        yield _data("debugger", {
            "issues": full_text,
            "pylint_raw": pylint_output,
            "files_checked": len(py_files) + len(js_files),
        })


async def _run_pylint(output_path) -> str:
    backend = output_path / "backend"
    if not backend.exists():
        return "No backend directory found."
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            ["python", "-m", "pylint", str(backend), "--output-format=text", "--score=no"],
            capture_output=True, text=True, timeout=30,
        )
        return (result.stdout + result.stderr).strip() or "No issues found."
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return "pylint not available or timed out."


def _status(agent: str, message: str) -> dict:
    return {"type": "status", "agent": agent, "message": message}

def _thought(agent: str, chunk: str) -> dict:
    return {"type": "thought", "agent": agent, "chunk": chunk}

def _data(agent: str, payload: dict) -> dict:
    return {"type": "data", "agent": agent, "data": payload}
