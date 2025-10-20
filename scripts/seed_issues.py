#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seed GitHub issues from a JSON spec.
Usage:
  export GITHUB_TOKEN=ghp_xxx
  python scripts/seed_issues.py --repo JCDemon/vdpt --input .github/seed/issues.json --assignee JCDemon
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def github_request(method: str, path: str, token: str, data: Optional[dict] = None) -> dict:
    api = "https://api.github.com"
    payload = None if data is None else json.dumps(data).encode("utf-8")
    req = Request(
        f"{api}{path}",
        data=payload,
        method=method.upper(),
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8") or "{}")
    except HTTPError as e:  # import HTTPError only if used
        msg = e.read().decode("utf-8") if hasattr(e, "read") else str(e)
        raise RuntimeError(f"GitHub API {method} {path} failed {e.code}: {msg}") from e


def get_milestone_number(owner, repo, title, token):
    for state in ("open", "closed"):
        path = f"/repos/{owner}/{repo}/milestones?state={state}&per_page=100"
        for m in github_request("GET", path, token=token):
            if m.get("title") == title:
                return m.get("number")
    raise RuntimeError(f"Milestone '{title}' not found in {owner}/{repo}")


def ensure_labels(owner, repo, labels, token):
    path = f"/repos/{owner}/{repo}/labels?per_page=100"
    existing = {label["name"] for label in github_request("GET", path, token=token)}
    create_path = f"/repos/{owner}/{repo}/labels"
    for name in labels:
        if name not in existing:
            # default light gray if not pre-created by metadata sync
            github_request(
                "POST",
                create_path,
                token=token,
                data={"name": name, "color": "ededed"},
            )


def create_issue(owner, repo, item, assignees, token):
    labels = item.get("labels", [])
    if labels:
        ensure_labels(owner, repo, labels, token)
    milestone_title = item.get("milestone")
    milestone_num = None
    if milestone_title:
        milestone_num = get_milestone_number(owner, repo, milestone_title, token)
    payload = {
        "title": item["title"],
        "body": item.get("body", ""),
        "labels": labels,
    }
    if assignees:
        payload["assignees"] = assignees
    if milestone_num is not None:
        payload["milestone"] = milestone_num
    return github_request(
        "POST",
        f"/repos/{owner}/{repo}/issues",
        token=token,
        data=payload,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, help="owner/repo")
    p.add_argument("--input", required=True, help="path to issues.json")
    p.add_argument("--assignee", action="append", default=[], help="assignee(s), can repeat")
    p.add_argument("--token", default=os.getenv("GITHUB_TOKEN"))
    args = p.parse_args()
    if not args.token:
        sys.exit("Missing token. Set GITHUB_TOKEN env or --token.")
    owner, repo = args.repo.split("/", 1)
    with open(args.input, "r", encoding="utf-8") as fh:
        items = json.load(fh)
    created = []
    for it in items:
        resp = create_issue(owner, repo, it, args.assignee, args.token)
        created.append(resp["html_url"])
    print("\n".join(created))


if __name__ == "__main__":
    main()
