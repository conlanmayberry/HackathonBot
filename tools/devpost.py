import httpx
from bs4 import BeautifulSoup


async def search_devpost(query: str, university: str = None, max_results: int = 10) -> list[dict]:
    """Search Devpost for hackathon projects matching query, optionally filtered by university."""
    params = {"search[q]": query, "challenge_type": "all"}
    if university:
        params["search[q]"] += f" {university}"

    headers = {"User-Agent": "Mozilla/5.0 (compatible; HackathonBot/1.0)"}

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            resp = await client.get(
                "https://devpost.com/software/search",
                params=params,
                headers=headers,
            )
            resp.raise_for_status()
        except httpx.HTTPError:
            return []

    soup = BeautifulSoup(resp.text, "html.parser")
    projects = []

    for card in soup.select("div.software-entry")[:max_results]:
        title_el = card.select_one("h5.software-entry-name a")
        desc_el = card.select_one("p.entry-body")
        tags_els = card.select("span.cp-tag")
        school_el = card.select_one("span.software-entry-school")

        if not title_el:
            continue

        projects.append({
            "title": title_el.get_text(strip=True),
            "url": title_el.get("href", ""),
            "description": desc_el.get_text(strip=True) if desc_el else "",
            "university": school_el.get_text(strip=True) if school_el else "",
            "tags": [t.get_text(strip=True) for t in tags_els],
        })

    return projects


async def search_devpost_winners(theme: str, max_results: int = 15) -> list[dict]:
    """Search specifically for winning/prize-winning hackathon projects."""
    return await search_devpost(f"{theme} winner prize", max_results=max_results)
