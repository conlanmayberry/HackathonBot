import re
import httpx

API_URL = "https://devpost.com/api/hackathons"
HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (compatible; HackathonBot/1.0)",
}


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


async def search_hackathons(query: str = "", status: str = None, max_results: int = 12) -> list[dict]:
    """Search Devpost for hackathons by free-text query (name or location).

    status: optional one of 'upcoming', 'open', 'ended'. None = all statuses.
    """
    params = {"search": query, "order_by": "relevance"}
    if status:
        params["status[]"] = status

    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        try:
            resp = await client.get(API_URL, params=params, headers=HEADERS)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return []

    hackathons = []
    for h in data.get("hackathons", [])[:max_results]:
        loc = h.get("displayed_location", {})
        hackathons.append({
            "title": h.get("title", ""),
            "url": h.get("url", ""),
            "deadline": h.get("submission_period_dates", ""),
            "location": loc.get("location", "Online") if isinstance(loc, dict) else "Online",
            "prize_amount": _strip_html(h.get("prize_amount", "")),
            "themes": [t.get("name", "") for t in h.get("themes", [])],
            "open_state": h.get("open_state", ""),
            "organization": h.get("organization_name", ""),
            "time_left": h.get("time_left_to_submission", ""),
        })

    return hackathons


async def lookup_hackathon(name: str) -> dict:
    """Look up a specific hackathon by name across past/present/future.

    Returns {"found": bool, "match": {...} | None, "candidates": [...]}.
    """
    results = await search_hackathons(query=name, status=None, max_results=8)
    if not results:
        return {"found": False, "match": None, "candidates": []}

    target = name.lower().strip()
    for h in results:
        title = h["title"].lower()
        # exact-ish match: target is a prefix/substring of the title or vice versa
        if target in title or title.startswith(target.split()[0] if target.split() else target):
            return {"found": True, "match": h, "candidates": results}

    return {"found": False, "match": None, "candidates": results}
