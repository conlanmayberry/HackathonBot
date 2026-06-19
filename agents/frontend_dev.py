import os
from typing import AsyncGenerator
import anthropic
from tools.file_writer import write_file

MODEL = "claude-opus-4-8"
SYSTEM = "You are an expert frontend developer. Write complete, working code — no placeholders, no TODOs. Every file must be immediately runnable."


class FrontendDevAgent:
    async def run(self, project_title: str, project_description: str, tech_stack: list[str], research_report: str, instructions: str = "") -> AsyncGenerator[dict, None]:
        yield _status("frontend_dev", f"Designing frontend for '{project_title}'…")

        extra = f"\n\nAdditional instructions from the team (follow these closely):\n{instructions}" if instructions else ""

        prompt = f"""Build the complete frontend for this hackathon project.

Project: {project_title}
Description: {project_description}
Tech stack: {', '.join(tech_stack)}
Research context:
{research_report[:800]}{extra}

Use vanilla HTML/CSS/JS unless the tech stack specifies React.

Return your response as FILE blocks in this exact format:
===FILE: relative/path/to/file.ext===
<file contents here>
===END===

Include at minimum: index.html, style.css, app.js
Make it look impressive for hackathon judges."""

        client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), timeout=120.0, max_retries=2)
        full_text = ""
        async with client.messages.stream(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for chunk in stream.text_stream:
                full_text += chunk
                yield _thought("frontend_dev", chunk)

        files_written = _parse_and_write(full_text, project_title, "frontend")
        yield _status("frontend_dev", f"Wrote {len(files_written)} frontend files.")
        yield _data("frontend_dev", {"files": files_written})


def _parse_and_write(raw: str, project_title: str, subdir: str) -> list[str]:
    written = []
    for part in raw.split("===FILE:")[1:]:
        header_end = part.find("===")
        if header_end == -1:
            continue
        rel_path = part[:header_end].strip()
        rest = part[header_end + 3:]
        end_marker = rest.find("===END===")
        content = rest[:end_marker].strip() if end_marker != -1 else rest.strip()
        dest = write_file(project_title, f"{subdir}/{rel_path}", content)
        written.append(dest)
    return written


def _status(agent: str, message: str) -> dict:
    return {"type": "status", "agent": agent, "message": message}

def _thought(agent: str, chunk: str) -> dict:
    return {"type": "thought", "agent": agent, "chunk": chunk}

def _data(agent: str, payload: dict) -> dict:
    return {"type": "data", "agent": agent, "data": payload}
