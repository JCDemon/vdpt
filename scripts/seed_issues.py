#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seed GitHub issues from a JSON spec.
Usage:
  export GITHUB_TOKEN=ghp_xxx
  python scripts/seed_issues.py --repo JCDemon/vdpt --input .github/seed/issues.json --assignee JCDemon
"""
from __future__ import annotations
import argparse, json, os, sys, urllib.request as req, urllib.error as err

API = "https://api.github.com"

def call(url, method="GET", data=None, token=None):
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "vdpt-seed-script/0.1",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if data is not None:
        data = json.dumps(data).encode("utf-8")
    r = req.Request(url, data=data, method=method, headers=headers)
    with req.urlopen(r) as resp:
        return json.loads(resp.read().decode("utf-8"))

def get_milestone_number(owner, repo, title, token):
    for state in ("open","closed"):
        url = f"{API}/repos/{owner}/{repo}/milestones?state={state}&per_page=100"
        for m in call(url, token=token):
            if m.get("title") == title:
                return m.get("number")
    raise RuntimeError(f"Milestone '{title}' not found in {owner}/{repo}")

def ensure_labels(owner, repo, labels, token):
    url = f"{API}/repos/{owner}/{repo}/labels?per_page=100"
    existing = {l["name"] for l in call(url, token=token)}
    create_url = f"{API}/repos/{owner}/{repo}/labels"
    for name in labels:
        if name not in existing:
            # default light gray if not pre-created by metadata sync
            call(create_url, method="POST", token=token, data={"name": name, "color": "ededed"})

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
    return call(f"{API}/repos/{owner}/{repo}/issues", method="POST", data=payload, token=token)

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
    items = json.load(open(args.input, "r", encoding="utf-8"))
    created = []
    for it in items:
        resp = create_issue(owner, repo, it, args.assignee, args.token)
        created.append(resp["html_url"])
    print("\n".join(created))

if __name__ == "__main__":
    main()
