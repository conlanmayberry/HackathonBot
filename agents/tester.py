import os
import asyncio
import subprocess
from typing import AsyncGenerator
import anthropic
from tools.file_writer import get_output_path, list_output_files, write_file

MODEL = "claude-opus-4-8"


class TesterAgent:
    async def run(self, project_title: str, project_description: str) -> AsyncGenerator[dict, None]:
        yield _status("tester", "Reading backend code to write targeted tests…")

        output_path = get_output_path(project_title)
        files = list_output_files(project_title)
        py_files = [f for f in files if f.endswith(".py") and "backend" in f]

        code_snippets = []
        for rel in py_files[:4]:
            full = output_path / rel
            try:
                content = full.read_text(encoding="utf-8")[:1500]
                code_snippets.append(f"### {rel}\n```python\n{content}\n```")
            except OSError:
                pass

        code_context = "\n\n".join(code_snippets) or "No backend code found."

        prompt = f"""You are a QA engineer writing tests for a hackathon project.

Project: {project_title}
Description: {project_description}

Backend code:
{code_context}

Write a complete pytest test file (test_backend.py) using FastAPI's TestClient.
- Import from the correct module path (assume backend/main.py contains the FastAPI app)
- Test the happy path for each route
- Test one error case per route
- Tests must be self-contained (no external services)

Return only the Python test code, no explanation or markdown fences."""

        client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), timeout=120.0, max_retries=2)
        full_text = ""
        async with client.messages.stream(
            model=MODEL,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for chunk in stream.text_stream:
                full_text += chunk
                yield _thought("tester", chunk)

        test_code = full_text.strip()
        if test_code.startswith("```"):
            test_code = test_code.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        write_file(project_title, "tests/test_backend.py", test_code)
        yield _status("tester", "Tests written. Running pytest…")

        pytest_output = await _run_pytest(output_path)

        yield _data("tester", {
            "test_file": str(output_path / "tests/test_backend.py"),
            "pytest_output": pytest_output,
        })


async def _run_pytest(output_path) -> str:
    tests_dir = output_path / "tests"
    if not tests_dir.exists():
        return "No tests directory found."
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            # -p no:cacheprovider keeps pytest from writing a .pytest_cache dir into output/
            ["python", "-m", "pytest", str(tests_dir), "-v", "--tb=short", "-p", "no:cacheprovider"],
            capture_output=True, text=True, timeout=60,
            cwd=str(output_path / "backend"),
        )
        return (result.stdout + result.stderr).strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return "pytest not available or timed out."


def _status(agent: str, message: str) -> dict:
    return {"type": "status", "agent": agent, "message": message}

def _thought(agent: str, chunk: str) -> dict:
    return {"type": "thought", "agent": agent, "chunk": chunk}

def _data(agent: str, payload: dict) -> dict:
    return {"type": "data", "agent": agent, "data": payload}
