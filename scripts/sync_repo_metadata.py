"""Synchronize GitHub labels and milestones for the vdpt repository.

This script creates or updates a set of labels and milestones on a target
GitHub repository.  It expects a personal access token with the appropriate
repository permissions supplied via the ``GITHUB_TOKEN`` environment variable
or ``--token`` CLI argument.

Example usage::

    python scripts/sync_repo_metadata.py --repo JCDemon/vdpt

Use ``--dry-run`` to preview the operations without calling the GitHub API.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

API_ROOT = "https://api.github.com"


@dataclass
class Label:
    name: str
    color: str
    description: Optional[str] = None

    @property
    def url_fragment(self) -> str:
        return quote(self.name, safe="")


@dataclass
class Milestone:
    title: str
    days_from_now: int
    description: Optional[str] = None


DEFAULT_LABELS: List[Label] = [
    Label("task", "1d76db"),
    Label("enhancement", "a2eeef"),
    Label("bug", "d73a4a"),
    Label("docs", "0e8a16", "Documentation improvements"),
    Label("design", "ff9f1c"),
    Label("infra", "6f42c1", "Infrastructure and tooling"),
    Label("research", "5319e7"),
    Label("good first issue", "7057ff", "Great for newcomers"),
    Label("help wanted", "008672"),
]

DEFAULT_MILESTONES: List[Milestone] = [
    Milestone("M1-MVP", 14, "Minimum viable product scope"),
    Milestone("M2-UX", 42, "User experience enhancements"),
    Milestone("M3-Study", 63, "Study and evaluation"),
    Milestone("M4-Paper", 84, "Publication preparation"),
]


def github_request(
    method: str,
    path: str,
    token: str,
    *,
    data: Optional[Dict] = None,
) -> Any:
    """Execute an authenticated GitHub API request."""
    url = f"{API_ROOT}{path}"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
    }
    payload = json.dumps(data).encode("utf-8") if data is not None else None
    request = Request(url, data=payload, headers=headers, method=method)
    with urlopen(request) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw) if raw else {}


def ensure_label(repo: str, label: Label, token: str, dry_run: bool) -> str:
    path = f"/repos/{repo}/labels/{label.url_fragment}"
    label_payload = {
        "new_name": label.name,
        "color": label.color,
    }
    if label.description is not None:
        label_payload["description"] = label.description

    if dry_run:
        return f"https://github.com/{repo}/labels/{label.url_fragment}"

    try:
        github_request("PATCH", path, token, data=label_payload)
    except HTTPError as error:
        if error.code != 404:
            raise
        create_payload = {
            "name": label.name,
            "color": label.color,
        }
        if label.description is not None:
            create_payload["description"] = label.description
        github_request("POST", f"/repos/{repo}/labels", token, data=create_payload)
    return f"https://github.com/{repo}/labels/{label.url_fragment}"


def ensure_milestone(repo: str, milestone: Milestone, token: str, dry_run: bool) -> str:
    due_date = _dt.datetime.now(_dt.timezone.utc) + _dt.timedelta(days=milestone.days_from_now)
    due_on = due_date.replace(hour=23, minute=59, second=59, microsecond=0)
    due_on_iso = due_on.isoformat().replace("+00:00", "Z")
    milestone_payload = {
        "title": milestone.title,
        "description": milestone.description or "",
        "due_on": due_on_iso,
        "state": "open",
    }

    if dry_run:
        return f"https://github.com/{repo}/milestone/{quote(milestone.title, safe='')}"

    # GitHub does not provide a direct lookup by title, so fetch all and match.
    existing = github_request("GET", f"/repos/{repo}/milestones", token, data=None)
    match = next((item for item in existing if item["title"] == milestone.title), None)

    if match:
        github_request(
            "PATCH",
            f"/repos/{repo}/milestones/{match['number']}",
            token,
            data=milestone_payload,
        )
        number = match["number"]
    else:
        created = github_request("POST", f"/repos/{repo}/milestones", token, data=milestone_payload)
        number = created["number"]

    return f"https://github.com/{repo}/milestone/{number}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="GitHub repository in the form owner/name")
    parser.add_argument(
        "--token",
        default=os.environ.get("GITHUB_TOKEN"),
        help="GitHub token (defaults to GITHUB_TOKEN env variable)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the generated URLs without performing API calls",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dry_run and not args.token:
        raise SystemExit("A GitHub token is required unless --dry-run is specified.")

    label_urls = [ensure_label(args.repo, label, args.token, args.dry_run) for label in DEFAULT_LABELS]
    milestone_urls = [ensure_milestone(args.repo, milestone, args.token, args.dry_run) for milestone in DEFAULT_MILESTONES]

    print("Labels:")
    for url in label_urls:
        print(f"  {url}")
    print("Milestones:")
    for url in milestone_urls:
        print(f"  {url}")


if __name__ == "__main__":
    main()
