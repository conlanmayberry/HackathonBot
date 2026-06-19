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
