"""QA / Verifier agent — the merged Debugger + Tester.

Responsibilities:
- Deterministic dependency reconciliation (imports → requirements.txt) so installs
  are complete and reliable.
- Write a robust pytest suite + conftest (once, in `setup`).
- Verify the project on every round (`verify`): pylint, pytest, and a static
  frontend-asset check (does index.html reference css/js that actually exist?).
- Report a structured pass/fail plus the *real* error output, which the planner
  feeds back to the frontend/backend devs so THEY fix their own code (build→verify→fix loop).

QA never edits the application code itself — fixing is the devs' job.
"""

import re
import ast
import sys
import asyncio
import subprocess
from typing import AsyncGenerator

from agents.llm import stream, MODEL
from tools.file_writer import get_output_path, list_output_files, write_file

# import-name -> PyPI package name, for the cases where they differ.
IMPORT_TO_PACKAGE = {
    "dotenv": "python-dotenv",
    "bs4": "beautifulsoup4",
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
    "jwt": "PyJWT",
    "github": "PyGithub",
    "dateutil": "python-dateutil",
    "fitz": "PyMuPDF",
    "serial": "pyserial",
    "OpenSSL": "pyOpenSSL",
    "Crypto": "pycryptodome",
    "multipart": "python-multipart",
    "jose": "python-jose",
    "socketio": "python-socketio",
}

CONFTEST = '''import os
import sys
from pathlib import Path

# Make the backend importable as `import main`, regardless of the pytest cwd.
BACKEND = Path(__file__).resolve().parent.parent / "backend"
if BACKEND.exists():
    sys.path.insert(0, str(BACKEND))

# Dummy secrets so importing the app never fails on a fresh checkout (tests mock
# the real clients).
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("GITHUB_TOKEN", "test-token")
'''


class QAAgent:
    # ── One-time setup: deps + conftest + write the test suite ──────────────
    async def setup(self, project_title: str, spec_text: str = "") -> AsyncGenerator[dict, None]:
        yield _status("qa", "Setting up QA — guaranteeing test dependencies and writing the test suite…")

        output_path = get_output_path(project_title)
        write_file(project_title, "tests/conftest.py", CONFTEST)
        test_deps = _ensure_test_deps(output_path)
        if test_deps:
            yield _status("qa", f"Added test dependencies to requirements.txt: {', '.join(test_deps)}")

        files = list_output_files(project_title)
        py_files = [f for f in files if f.endswith(".py") and f.startswith("backend")]
        code_snippets = []
        for rel in py_files[:8]:
            try:
                content = (output_path / rel).read_text(encoding="utf-8")[:4000]
                code_snippets.append(f"### {rel}\n```python\n{content}\n```")
            except OSError:
                pass
        code_context = "\n\n".join(code_snippets) or "No backend code found."
        spec_block = f"\n\nThe project's intended API contract:\n{spec_text[:2500]}\n" if spec_text else ""

        prompt = f"""You are a QA engineer writing a real, runnable pytest suite for a hackathon backend.

PROJECT: {project_title}
{spec_block}
Backend code:
{code_context}

Write a complete test file. Hard requirements:
- `from main import app` then `client = TestClient(app)` (a conftest.py already puts the
  backend on sys.path and sets dummy API keys — assume it exists).
- Cover EVERY endpoint: a happy path AND at least one error/edge case each (404 for unknown
  ids, 422 for invalid bodies, etc.).
- Pass with NO network access and NO real keys. If the backend calls an external service
  (Anthropic, OpenAI, requests, httpx to a third party), MOCK it with unittest.mock.patch
  against the symbol as imported in `main` (e.g. patch("main.client")). Inspect the imports
  above to patch the correct name.
- Pure pytest + fastapi.testclient + unittest.mock. No external test services.

Return ONLY the Python test file contents — no markdown fences, no prose."""

        full_text = ""
        async for kind, delta in stream(
            model=MODEL, max_tokens=16000, thinking=True, effort="high",
            messages=[{"role": "user", "content": prompt}],
        ):
            if kind == "text":
                full_text += delta
            yield _thought("qa", delta)

        write_file(project_title, "tests/test_backend.py", _strip_fence(full_text.strip()))
        yield _status("qa", "Test suite + conftest written.")
        yield _data("qa", {"test_deps_added": test_deps})

    # ── Per-round verification: deps + pylint + pytest + frontend assets ────
    async def verify(self, project_title: str) -> AsyncGenerator[dict, None]:
        output_path = get_output_path(project_title)
        files = list_output_files(project_title)
        py_files = [f for f in files if f.endswith(".py")]

        # Re-reconcile deps every round (a fix may have introduced a new import).
        deps_added = _reconcile_requirements(output_path, py_files)
        if deps_added:
            yield _status("qa", f"Reconciled requirements.txt (+{', '.join(deps_added)})")

        yield _status("qa", "Running pylint + pytest + frontend asset check…")
        pylint_output = await _run_pylint(output_path, py_files)
        pytest_code, pytest_output = await _run_pytest(output_path)
        missing_assets = _check_frontend_assets(output_path / "frontend")

        pylint_errors = re.findall(r":\s*([EF]\d{4}):", pylint_output)
        # pytest exit codes: 0 = all passed, 5 = no tests collected, None = unavailable.
        pytest_ok = pytest_code in (0, None)
        frontend_ok = not missing_assets
        passed = pytest_ok and not pylint_errors and frontend_ok

        backend_failures = ""
        if pylint_errors or (pytest_code not in (0, 5, None)):
            err_lines = [l for l in pylint_output.splitlines() if re.search(r":\s*[EF]\d{4}:", l)]
            backend_failures = (
                ("pylint errors:\n" + "\n".join(err_lines) + "\n\n" if err_lines else "")
                + ("pytest output:\n" + pytest_output[-2500:] if pytest_code not in (0, 5, None) else "")
            ).strip()

        frontend_failures = ""
        if missing_assets:
            frontend_failures = (
                "index.html references these local files that DO NOT EXIST in frontend/. "
                "Create them (or stop referencing them): " + ", ".join(missing_assets)
            )

        # Human-readable summary for the Debug/Tests panel.
        summary = []
        summary.append("✅ pytest passed" if pytest_ok else f"❌ pytest failed (exit {pytest_code})")
        summary.append("✅ no pylint errors" if not pylint_errors else f"❌ {len(pylint_errors)} pylint error(s)")
        summary.append("✅ frontend assets present" if frontend_ok else f"❌ missing frontend files: {', '.join(missing_assets)}")
        issues = "## QA results\n" + "\n".join(f"- {s}" for s in summary)

        yield _data("qa", {
            "passed": passed,
            "issues": issues,
            "pylint": pylint_output,
            "pytest_output": pytest_output,
            "backend_failures": backend_failures,
            "frontend_failures": frontend_failures,
            "files_checked": len(py_files),
            "deps_added": deps_added,
        })


# ── Dependency reconciliation ────────────────────────────────────────────────
def _reconcile_requirements(output_path, py_files: list[str]) -> list[str]:
    backend = output_path / "backend"
    backend_py = [output_path / f for f in py_files if f.startswith("backend")]
    if not backend.exists():
        return []

    local_modules = {p.stem for p in backend_py}
    imported = _third_party_imports(backend_py, local_modules)
    if not imported:
        return []
    if "fastapi" in imported:
        imported.add("uvicorn[standard]")

    req_path = backend / "requirements.txt"
    existing_lines, have = [], set()
    if req_path.exists():
        existing_lines = req_path.read_text(encoding="utf-8").splitlines()
        have = {_req_base(l) for l in existing_lines if l.strip() and not l.strip().startswith("#")}

    added = []
    for pkg in sorted(imported):
        if _req_base(pkg) not in have:
            added.append(pkg)
            have.add(_req_base(pkg))
    if added:
        lines = [l for l in existing_lines if l.strip()] + added
        req_path.parent.mkdir(parents=True, exist_ok=True)
        req_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return added


def _third_party_imports(py_paths, local_modules: set[str]) -> set[str]:
    stdlib = set(getattr(sys, "stdlib_module_names", set()))
    found: set[str] = set()
    for path in py_paths:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    found.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    found.add(node.module.split(".")[0])
    third: set[str] = set()
    for mod in found:
        if not mod or mod in stdlib or mod in local_modules:
            continue
        third.add(IMPORT_TO_PACKAGE.get(mod, mod))
    return third


def _ensure_test_deps(output_path) -> list[str]:
    req_path = output_path / "backend" / "requirements.txt"
    needed = {"pytest": "pytest>=8.0.0", "httpx": "httpx>=0.27.0"}
    existing, have = [], set()
    if req_path.exists():
        existing = req_path.read_text(encoding="utf-8").splitlines()
        have = {_req_base(l) for l in existing if l.strip() and not l.strip().startswith("#")}
    added = []
    for base, line in needed.items():
        if base not in have:
            added.append(base)
            existing.append(line)
    if added:
        req_path.parent.mkdir(parents=True, exist_ok=True)
        req_path.write_text("\n".join(l for l in existing if l.strip()) + "\n", encoding="utf-8")
    return added


def _req_base(line: str) -> str:
    name = line.strip().lower()
    for sep in ("==", ">=", "<=", "~=", ">", "<", "!=", "[", " ", ";"):
        idx = name.find(sep)
        if idx != -1:
            name = name[:idx]
    return name.strip()


# ── Frontend static check ────────────────────────────────────────────────────
def _check_frontend_assets(frontend_dir) -> list[str]:
    """Return css/js files that index.html references but that don't exist."""
    index = frontend_dir / "index.html"
    if not index.exists():
        return []
    try:
        html = index.read_text(encoding="utf-8")
    except OSError:
        return []
    missing = []
    # Match href/src with or without quotes.
    refs = re.findall(r'(?:href|src)\s*=\s*(?:"([^"]+)"|\'([^\']+)\'|([^\s">]+))', html)
    for quoted, single, unquoted in refs:
        ref = quoted or single or unquoted
        r = ref.strip()
        if r.startswith(("http://", "https://", "//", "data:", "#", "mailto:", "javascript:")):
            continue
        clean = r.split("?")[0].split("#")[0].lstrip("/")
        if clean.lower().endswith((".css", ".js")) and not (frontend_dir / clean).exists():
            missing.append(ref)
    return missing


def _strip_fence(code: str) -> str:
    if code.startswith("```"):
        first_nl = code.find("\n")
        if first_nl != -1:
            code = code[first_nl + 1:]
        if code.rstrip().endswith("```"):
            code = code.rstrip()[:-3]
    return code.strip()


# ── Subprocess runners ───────────────────────────────────────────────────────
async def _run_pylint(output_path, files: list[str]) -> str:
    # Lint the explicit .py files (NOT the directory): pointing pylint at a folder
    # without __init__.py triggers a spurious F0010 parse-error. Disable C/R (style/
    # refactor) and E0401 (import-error is env-dependent; the dep audit governs that).
    targets = [str(output_path / f) for f in files if f.startswith("backend") and f.endswith(".py")]
    if not targets:
        return "No backend Python files found."
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            ["python", "-m", "pylint", *targets, "--output-format=text", "--score=no",
             "--disable=C,R,E0401", "--max-line-length=120"],
            capture_output=True, text=True, timeout=45,
        )
        return (result.stdout + result.stderr).strip() or "No issues found."
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return "pylint not available or timed out."


async def _run_pytest(output_path):
    """Return (returncode_or_None, output_text). None = pytest unavailable."""
    tests_dir = output_path / "tests"
    if not tests_dir.exists():
        return None, "No tests directory found."
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            ["python", "-m", "pytest", "tests", "-v", "--tb=short", "-p", "no:cacheprovider"],
            capture_output=True, text=True, timeout=90, cwd=str(output_path),
        )
        return result.returncode, (result.stdout + result.stderr).strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None, "pytest not available or timed out."


def _status(agent: str, message: str, data: dict = None) -> dict:
    return {"type": "status", "agent": agent, "message": message, "data": data or {}}

def _thought(agent: str, chunk: str) -> dict:
    return {"type": "thought", "agent": agent, "chunk": chunk}

def _data(agent: str, payload: dict) -> dict:
    return {"type": "data", "agent": agent, "data": payload}
