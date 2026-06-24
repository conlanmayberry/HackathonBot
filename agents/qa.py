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
import hashlib
import asyncio
import subprocess
from typing import AsyncGenerator

from agents.llm import stream, MODEL_CODE
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

        # Writing a TestClient suite against code that's already in front of the model is
        # well-scoped work — Sonnet 4.6 at medium effort does it reliably for ~40% less.
        full_text = ""
        async for kind, delta in stream(
            model=MODEL_CODE, max_tokens=16000, thinking=True, effort="medium",
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

        yield _status("qa", "Verifying: pylint + pytest + the integration graph (assets, JS imports, API contract)…")
        pylint_output = await _run_pylint(output_path, py_files)
        pytest_code, pytest_output = await _run_pytest(output_path)

        # ── Integration checks: the failure class unit tests never catch — one
        # agent referencing an artifact another agent never produced. ──────────
        frontend_dir = output_path / "frontend"
        missing_assets = _check_frontend_assets(frontend_dir)      # HTML → css/js/asset refs
        broken_imports = _check_js_imports(frontend_dir)           # JS  → ES-module imports
        contract_gaps = _check_api_contract(output_path)           # frontend fetch ↔ backend routes

        pylint_errors = re.findall(r":\s*([EF]\d{4}):", pylint_output)
        # pytest exit codes: 0 = all passed, 5 = no tests collected, None = unavailable.
        pytest_ok = pytest_code in (0, None)
        frontend_ok = not (missing_assets or broken_imports)
        contract_ok = not contract_gaps
        static_clean = pytest_ok and not pylint_errors and frontend_ok and contract_ok

        # ── Runtime browser check: launch the real app and assert the page renders ──
        # This is the slowest, most expensive check by far (pip install + two servers + a
        # headless browser), so only run it once the cheap static checks are clean. If
        # pylint/pytest or the integration graph already failed, those failures route to the
        # devs and we re-verify next round anyway — and a build with a broken import or a
        # missing route won't render in a browser regardless. Skipping here saves a full
        # install+launch cycle on every failing round of the fix loop.
        runtime_skipped = not static_clean
        if static_clean:
            yield _status("qa", "Static checks clean — runtime browser check: starting app and loading in headless Chromium…")
            runtime_problems = await _check_runtime(output_path)
        else:
            yield _status("qa", "Skipping the runtime browser check until pylint/pytest/integration checks are clean.")
            runtime_problems = []
        runtime_ok = not runtime_problems
        passed = static_clean and runtime_ok

        backend_failures = ""
        if pylint_errors or (pytest_code not in (0, 5, None)):
            err_lines = [l for l in pylint_output.splitlines() if re.search(r":\s*[EF]\d{4}:", l)]
            backend_failures = (
                ("pylint errors:\n" + "\n".join(err_lines) + "\n\n" if err_lines else "")
                + ("pytest output:\n" + pytest_output[-2500:] if pytest_code not in (0, 5, None) else "")
            ).strip()
        if contract_gaps:
            # The frontend calls endpoints the backend doesn't serve. Per the build
            # spec the backend owns the contract, so route the fix there.
            gap_lines = "\n".join(f"- frontend calls `{m} {p}` but no backend route matches it" for m, p in contract_gaps)
            backend_failures = (backend_failures + "\n\n" if backend_failures else "") + (
                "API CONTRACT GAP — the frontend calls these endpoints that the backend does NOT "
                "implement. Add the missing routes (matching the build spec's contract), or if the "
                "frontend is calling the wrong path, the path it expects is shown:\n" + gap_lines
            )

        frontend_failures = ""
        fe_problems = []
        if missing_assets:
            fe_problems.append(
                "HTML references these local files that DO NOT EXIST — create them (or stop "
                "referencing them): " + ", ".join(missing_assets))
        if broken_imports:
            fe_problems.append(
                "These JS module imports do NOT resolve to a file on disk — a single failed top-level "
                "import takes down the whole script, so the page never boots. Create the missing files "
                "(or fix the paths):\n" + "\n".join(f"- {i}" for i in broken_imports))
        if runtime_problems:
            fe_problems.append(
                "RUNTIME CHECK FAILED — the app was launched and loaded in a real browser but these "
                "problems were detected. Fix the rendering or server startup issues:\n"
                + "\n".join(f"- {p}" for p in runtime_problems))
        if fe_problems:
            frontend_failures = "\n\n".join(fe_problems)

        # Human-readable summary for the Debug/Tests panel.
        summary = [
            "✅ pytest passed" if pytest_ok else f"❌ pytest failed (exit {pytest_code})",
            "✅ no pylint errors" if not pylint_errors else f"❌ {len(pylint_errors)} pylint error(s)",
            "✅ frontend assets resolve" if not missing_assets else f"❌ missing assets: {', '.join(missing_assets)}",
            "✅ JS imports resolve" if not broken_imports else f"❌ {len(broken_imports)} unresolved JS import(s)",
            "✅ frontend↔backend API contract aligns" if contract_ok else f"❌ {len(contract_gaps)} API contract gap(s)",
            "⏭️ runtime browser check skipped (clear the checks above first)" if runtime_skipped
            else ("✅ runtime browser check passed" if runtime_ok else f"❌ {len(runtime_problems)} runtime problem(s)"),
        ]
        issues = "## QA results\n" + "\n".join(f"- {s}" for s in summary)

        yield _data("qa", {
            "passed": passed,
            "issues": issues,
            "pylint": pylint_output,
            "pytest_output": pytest_output,
            "backend_failures": backend_failures,
            "frontend_failures": frontend_failures,
            "runtime_problems": runtime_problems,
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


def _check_js_imports(frontend_dir) -> list[str]:
    """Resolve every relative ES-module import in the frontend's JS and report the
    ones that don't point at a real file. A single failed top-level import silently
    bricks the whole page, so this is one of the highest-value integration checks."""
    if not frontend_dir.exists():
        return []
    js_files = [p for p in frontend_dir.rglob("*.js") if p.is_file()]
    # static `import ... from '<path>'`, side-effect `import '<path>'`, re-export
    # `export ... from '<path>'`, and dynamic `import('<path>')`.
    pat = re.compile(
        r"""(?:import|export)\s+(?:[^'"]*?\sfrom\s+)?['"]([^'"]+)['"]"""
        r"""|import\s*\(\s*['"]([^'"]+)['"]\s*\)"""
    )
    problems = []
    for js in js_files:
        try:
            src = js.read_text(encoding="utf-8")
        except OSError:
            continue
        for m in pat.finditer(src):
            spec = (m.group(1) or m.group(2) or "").strip()
            # Only verify local/relative specifiers (bare specifiers are CDN/package imports).
            if not spec.startswith((".", "/")):
                continue
            spec_clean = spec.split("?")[0].split("#")[0]
            base = (js.parent / spec_clean).resolve()
            candidates = [base]
            if not base.suffix:  # extensionless import → try .js and /index.js
                candidates += [base.with_suffix(".js"), base / "index.js"]
            if not any(c.exists() for c in candidates):
                rel = js.relative_to(frontend_dir).as_posix()
                problems.append(f"{rel} imports '{spec}' which does not exist")
    return problems


# Methods we look for on both sides of the contract.
_HTTP_METHODS = ("get", "post", "put", "patch", "delete")


def _check_api_contract(output_path) -> list[tuple[str, str]]:
    """Cross-check the frontend's HTTP calls against the backend's routes.

    Returns (method, path) pairs the frontend calls that no backend route serves.
    Conservative by design — only flags calls whose path is statically determinable,
    so dynamic/templated URLs never produce false positives that thrash the fix loop."""
    backend = output_path / "backend"
    frontend = output_path / "frontend"
    if not backend.exists() or not frontend.exists():
        return []

    backend_routes = _backend_routes(backend)
    if not backend_routes:
        return []  # can't extract routes → don't guess
    backend_norm = {(meth, _norm_path(path)) for meth, path in backend_routes}
    backend_norm_paths = {p for _, p in backend_norm}

    gaps = []
    seen = set()
    for js in [p for p in frontend.rglob("*.js") if p.is_file()]:
        try:
            src = js.read_text(encoding="utf-8")
        except OSError:
            continue
        for meth, path in _frontend_calls(src):
            # Skip calls to external services — only our own backend is the contract.
            if re.match(r"https?://", path) and not re.match(r"https?://(127\.0\.0\.1|localhost)", path):
                continue
            norm = _norm_path(path)
            if not norm.startswith("/"):
                continue
            segs = [s for s in norm.split("/") if s]
            if segs and segs[0] == "{}":
                continue  # dynamic first segment — can't anchor a comparison, skip (no false positives)
            key = (meth, norm)
            if key in seen:
                continue
            seen.add(key)
            if norm not in backend_norm_paths:
                gaps.append((meth.upper() if meth != "?" else "GET?", path))
    return gaps


def _backend_routes(backend_dir) -> list[tuple[str, str]]:
    routes = []
    deco = re.compile(r"@\w+\.(" + "|".join(_HTTP_METHODS) + r")\s*\(\s*['\"]([^'\"]+)['\"]")
    for py in backend_dir.rglob("*.py"):
        try:
            src = py.read_text(encoding="utf-8")
        except OSError:
            continue
        for m in deco.finditer(src):
            routes.append((m.group(1), m.group(2)))
    return routes


def _frontend_calls(src: str) -> list[tuple[str, str]]:
    """Extract (method, path) pairs from fetch()/axios calls with a statically
    determinable path. Conservative: template literals are kept only when a leading
    base-url/origin can be stripped to leave a path that starts with '/'."""
    calls = []
    # fetch('/x') / fetch("/x") — single/double-quoted static strings.
    for m in re.finditer(r"""fetch\s*\(\s*(['"])([^'"]+)\1""", src):
        calls.append(("?", m.group(2)))
    # fetch(`...`) — template literal: strip a leading ${BASE}/origin, keep if path-rooted.
    for m in re.finditer(r"fetch\s*\(\s*`([^`]+)`", src):
        body = re.sub(r"^\$\{[^}]+\}", "", m.group(1))
        body = re.sub(r"^https?://[^/]+", "", body)
        if body.startswith("/"):
            calls.append(("?", body))
    # axios.get('/x'), api.post("/x"), client.delete('/x') — single/double-quoted.
    for m in re.finditer(r"""\.(""" + "|".join(_HTTP_METHODS) + r""")\s*\(\s*(['"])([^'"]+)\2""", src):
        calls.append((m.group(1), m.group(3)))
    return calls


def _norm_path(url: str) -> str:
    """Reduce a URL to a comparable path template: strip origin/base, query, and
    collapse path params (`/items/123`, `/items/${id}`, `/items/{id}`) to `/items/{}`."""
    u = url.strip()
    # Strip a leading origin (http://host:port) if present.
    u = re.sub(r"^https?://[^/]+", "", u)
    # Strip a known base-url variable prefix like ${API_BASE} left at the head.
    u = re.sub(r"^\$\{[^}]+\}", "", u)
    u = u.split("?")[0].split("#")[0]
    if not u.startswith("/"):
        u = "/" + u if u and not u.startswith("$") else u
    # Collapse dynamic / param segments to a single placeholder.
    segs = []
    for seg in u.split("/"):
        if not seg:
            segs.append(seg)
            continue
        if seg.startswith(("{", ":", "$")) or re.fullmatch(r"\d+", seg) or "${" in seg:
            segs.append("{}")
        else:
            segs.append(seg)
    norm = "/".join(segs)
    return norm.rstrip("/") or "/"


# Hashes of requirements.txt files we've already pip-installed successfully this process,
# so the fix loop doesn't reinstall unchanged deps on every verify round.
_INSTALLED_REQS: set[str] = set()


async def _check_runtime(output_path, backend_port: int = 18100, frontend_port: int = 18200) -> list[str]:
    """Launch the app in subprocesses, open index.html in headless Chromium, and assert the
    page renders real content with no console errors and no failed API requests.
    Returns a list of problem strings; empty = all clear.
    Skips cleanly (returns []) when playwright is not installed or the project layout
    doesn't have the expected backend/frontend dirs."""
    try:
        from playwright.async_api import async_playwright  # noqa: PLC0415
    except ImportError:
        return []

    backend_dir = output_path / "backend"
    frontend_dir = output_path / "frontend"
    if not backend_dir.exists() or not (frontend_dir / "index.html").exists():
        return []

    # Locate the FastAPI entrypoint (main.py is the canonical name from the build spec).
    main_module = next(
        (stem for stem in ("main", "app", "server") if (backend_dir / f"{stem}.py").exists()),
        None,
    )
    if not main_module:
        return ["runtime: cannot find main.py / app.py / server.py in backend/"]

    problems: list[str] = []
    backend_proc = None
    frontend_proc = None

    try:
        # Install backend deps so the server can actually start. The fix loop calls verify()
        # several times; skip the (slow) reinstall when requirements.txt is byte-identical to
        # a successful install we already did this process — but reinstall if QA changed it.
        req = backend_dir / "requirements.txt"
        if req.exists():
            req_hash = hashlib.md5(req.read_bytes()).hexdigest()
            if req_hash not in _INSTALLED_REQS:
                pip = await asyncio.create_subprocess_exec(
                    "python", "-m", "pip", "install", "-r", "requirements.txt", "-q",
                    cwd=str(backend_dir),
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                try:
                    await asyncio.wait_for(pip.wait(), timeout=120.0)
                except asyncio.TimeoutError:
                    pip.kill()
                    return ["runtime: pip install timed out — deps not installed"]
                if pip.returncode == 0:
                    _INSTALLED_REQS.add(req_hash)  # only cache a clean install

        # Launch backend.
        backend_proc = await asyncio.create_subprocess_exec(
            "python", "-m", "uvicorn", f"{main_module}:app",
            "--host", "127.0.0.1", "--port", str(backend_port),
            cwd=str(backend_dir),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )

        # Launch frontend (simple static file server).
        frontend_proc = await asyncio.create_subprocess_exec(
            "python", "-m", "http.server", str(frontend_port),
            cwd=str(frontend_dir),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )

        # Poll until the backend is accepting connections (up to 15 s).
        backend_ready = False
        for _ in range(30):
            await asyncio.sleep(0.5)
            try:
                _, writer = await asyncio.wait_for(
                    asyncio.open_connection("127.0.0.1", backend_port), timeout=1.0
                )
                writer.close()
                backend_ready = True
                break
            except Exception:
                pass

        if not backend_ready:
            return [f"runtime: backend failed to start on port {backend_port} within 15 s"]

        await asyncio.sleep(0.5)  # give frontend http.server a moment

        # ── Headless browser smoke test ───────────────────────────────────────
        console_errors: list[str] = []
        failed_requests: list[str] = []

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            page = await browser.new_page()

            page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
            page.on("requestfailed", lambda req: failed_requests.append(f"FAILED {req.url}"))

            async def _on_response(resp) -> None:
                if resp.status >= 400 and f":{backend_port}" in resp.url:
                    failed_requests.append(f"HTTP {resp.status} {resp.url}")
            page.on("response", _on_response)

            try:
                await page.goto(
                    f"http://127.0.0.1:{frontend_port}/index.html",
                    wait_until="networkidle",
                    timeout=20_000,
                )
            except Exception as exc:
                await browser.close()
                return [f"runtime: page failed to load — {exc}"]

            body_text = (await page.inner_text("body")).strip()
            await browser.close()

        # Report browser console errors (cap at 5 to keep the repair prompt focused).
        problems.extend(f"runtime: browser console error: {e}" for e in console_errors[:5])
        problems.extend(f"runtime: {r}" for r in failed_requests[:5])

        # Detect "stuck on loading" — body either empty or only contains a spinner.
        loading_words = {"loading", "fetching", "please wait", "spinner"}
        is_stuck = len(body_text) < 50 or any(w in body_text.lower() for w in loading_words)
        if is_stuck:
            preview = body_text[:200].replace("\n", " ")
            problems.append(
                f"runtime: page appears empty or stuck on a loading state — body preview: {preview!r}"
            )

        return problems

    except Exception as exc:  # noqa: BLE001
        return [f"runtime: browser check crashed unexpectedly — {exc}"]

    finally:
        for proc in (backend_proc, frontend_proc):
            if proc and proc.returncode is None:
                proc.kill()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                except (asyncio.TimeoutError, Exception):
                    pass


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
