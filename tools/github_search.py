import os
from github import Github, Auth


def _client() -> Github:
    token = os.getenv("GITHUB_TOKEN")
    # A short timeout + capped retries so a slow/rate-limited API can't hang the run.
    if token:
        return Github(auth=Auth.Token(token), timeout=10, retry=1)
    return Github(timeout=10, retry=1)


def search_github(query: str, language: str = None, max_results: int = 10) -> list[dict]:
    """Search GitHub repositories matching query.

    Returns [] on any error (network, auth, rate limit) so callers never block.
    """
    search_query = query
    if language:
        search_query += f" language:{language}"

    try:
        g = _client()
        results = g.search_repositories(query=search_query, sort="stars", order="desc")
        repos = []
        for repo in results[:max_results]:
            # Pull everything from the already-loaded search payload — no extra
            # per-repo API calls (get_topics() was the previous hang/N+1 source).
            try:
                topics = repo.raw_data.get("topics", []) if repo.raw_data else []
            except Exception:
                topics = []
            repos.append({
                "name": repo.full_name,
                "url": repo.html_url,
                "description": repo.description or "",
                "stars": repo.stargazers_count,
                "topics": topics,
                "language": repo.language or "",
            })
        return repos
    except Exception:
        # Broad on purpose: GithubException, rate limits, timeouts, connection errors.
        return []


def search_github_hackathon(project_idea: str, max_results: int = 10) -> list[dict]:
    """Search GitHub for hackathon projects similar to the given idea."""
    return search_github(f"{project_idea} hackathon", max_results=max_results)
