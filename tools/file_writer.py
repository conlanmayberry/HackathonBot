import re
from pathlib import Path


OUTPUT_DIR = Path("output")

# Directories and file patterns that are build/runtime artifacts, never project files.
IGNORE_DIRS = {"__pycache__", ".pytest_cache", "node_modules", ".git", ".venv", "venv", ".mypy_cache", ".ruff_cache", "dist", "build", ".idea", ".vscode"}
IGNORE_SUFFIXES = {".pyc", ".pyo", ".log"}


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:50] or "project"


# Backwards-compatible alias.
_slug = slugify


def get_output_path(project_title: str) -> Path:
    return OUTPUT_DIR / slugify(project_title)


def write_file(project_title: str, relative_path: str, content: str) -> str:
    """Write content to output/<slug>/<relative_path>, creating dirs as needed."""
    dest = get_output_path(project_title) / relative_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content, encoding="utf-8")
    return str(dest)


def _is_ignored(path: Path, base: Path) -> bool:
    rel_parts = path.relative_to(base).parts
    if any(part in IGNORE_DIRS for part in rel_parts):
        return True
    if path.suffix.lower() in IGNORE_SUFFIXES:
        return True
    return False


def iter_project_files(base: Path) -> list[Path]:
    """Absolute paths of real project files under base, excluding build artifacts."""
    if not base.exists():
        return []
    return [p for p in base.rglob("*") if p.is_file() and not _is_ignored(p, base)]


def list_output_files(project_title: str) -> list[str]:
    """Relative paths of real project files for a project (no cache/build junk)."""
    base = get_output_path(project_title)
    return [str(p.relative_to(base)) for p in iter_project_files(base)]


def parse_file_blocks(raw: str) -> list[tuple[str, str]]:
    """Parse an LLM response containing one or more file blocks of the form:

        ===FILE: relative/path/to/file.ext===
        <file contents>
        ===END===

    Returns a list of (relative_path, content) tuples. Robust to a missing
    trailing ===END===, to a stray ```lang fence wrapping the contents, and to
    any prose the model emits before the first block.
    """
    blocks: list[tuple[str, str]] = []
    for part in raw.split("===FILE:")[1:]:
        header_end = part.find("===")
        if header_end == -1:
            continue
        rel_path = part[:header_end].strip().lstrip("/").replace("\\", "/")
        rest = part[header_end + 3:]
        end_marker = rest.find("===END===")
        content = rest[:end_marker] if end_marker != -1 else rest
        content = _strip_code_fence(content).strip("\n")
        if rel_path:
            blocks.append((rel_path, content))
    return blocks


def _strip_code_fence(text: str) -> str:
    """Remove a single wrapping ```lang ... ``` fence if the model added one."""
    stripped = text.strip()
    if stripped.startswith("```"):
        first_nl = stripped.find("\n")
        if first_nl != -1:
            stripped = stripped[first_nl + 1:]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
        return stripped
    return text


def write_file_blocks(project_title: str, subdir: str, raw: str) -> list[str]:
    """Parse file blocks from ``raw`` and write each under output/<slug>/<subdir>/."""
    written = []
    for rel_path, content in parse_file_blocks(raw):
        target = f"{subdir}/{rel_path}" if subdir else rel_path
        written.append(write_file(project_title, target, content))
    return written
