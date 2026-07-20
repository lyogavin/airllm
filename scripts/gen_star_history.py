#!/usr/bin/env python3
"""Generate a star-history chart for a GitHub repo, using the authenticated GitHub API.

Unlike the anonymous star-history.com SVG API (which is rate-limited and returns an
empty chart for large repos), this samples stargazer timestamps directly with a token,
so it works reliably for repos with tens of thousands of stars. Output is a PNG, which
always renders in GitHub markdown.

Usage:
    GITHUB_TOKEN=... python scripts/gen_star_history.py [owner/repo] [output.png] [theme]

theme is "light" (default) or "dark".
"""
import datetime
import json
import math
import os
import sys
import urllib.error
import urllib.request

REPO = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("REPO", "lyogavin/airllm")
OUT = sys.argv[2] if len(sys.argv) > 2 else "assets/star-history.png"
THEME = (sys.argv[3] if len(sys.argv) > 3 else os.environ.get("THEME", "light")).lower()
TOKEN = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
PER_PAGE = 100
MAX_SAMPLES = 30

THEMES = {
    "light": {"bg": "#ffffff", "fg": "#24292f", "grid": "#d0d7de", "line": "#e34a4a"},
    "dark": {"bg": "#0d1117", "fg": "#c9d1d9", "grid": "#30363d", "line": "#ff6b6b"},
}


def gh(url, accept="application/vnd.github+json"):
    headers = {"Accept": accept, "User-Agent": "airllm-star-history"}
    if TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            msg = json.loads(body).get("message", body)
        except json.JSONDecodeError:
            msg = body
        hint = ""
        if e.code == 403 and "stargazers" in url:
            hint = (
                "\nGitHub restricts /stargazers to admins/collaborators; the "
                "Actions GITHUB_TOKEN cannot access it. Use a collaborator PAT "
                "via the STAR_HISTORY_TOKEN secret."
            )
        raise SystemExit(f"GitHub API {e.code} for {url}: {msg}{hint}") from e


def main():
    info = gh(f"https://api.github.com/repos/{REPO}")
    total = int(info["stargazers_count"])
    if total <= 0:
        raise SystemExit("repo has no stars")

    max_page = max(1, math.ceil(total / PER_PAGE))
    if max_page == 1:
        pages = [1]
    else:
        n = min(MAX_SAMPLES, max_page)
        pages = sorted({1, max_page} | {
            round(1 + i * (max_page - 1) / (n - 1)) for i in range(n)
        })

    points = []
    for p in pages:
        data = gh(
            f"https://api.github.com/repos/{REPO}/stargazers?per_page={PER_PAGE}&page={p}",
            accept="application/vnd.github.star+json",
        )
        if not data:
            continue
        starred_at = data[0]["starred_at"]
        cumulative = (p - 1) * PER_PAGE + 1
        dt = datetime.datetime.fromisoformat(starred_at.replace("Z", "+00:00"))
        points.append((dt, cumulative))

    points.append((datetime.datetime.now(datetime.timezone.utc), total))
    points = sorted(set(points))
    if len(points) < 2:
        raise SystemExit("not enough data points to plot")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    xs = [d for d, _ in points]
    ys = [c for _, c in points]

    c = THEMES.get(THEME, THEMES["light"])
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(c["bg"])
    ax.set_facecolor(c["bg"])

    ax.plot(xs, ys, color=c["line"], linewidth=2.5, marker="o", markersize=4)
    ax.fill_between(xs, ys, color=c["line"], alpha=0.10)
    ax.set_title(f"Star History — {REPO}", fontsize=15, color=c["fg"])
    ax.set_ylabel("GitHub Stars", fontsize=12, color=c["fg"])
    ax.set_xlabel("Date", fontsize=12, color=c["fg"])
    ax.grid(True, alpha=0.3, color=c["grid"])
    ax.tick_params(colors=c["fg"])
    for spine in ax.spines.values():
        spine.set_color(c["grid"])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    fig.savefig(OUT, dpi=130, facecolor=c["bg"])
    print(f"wrote {OUT} ({THEME}): {len(points)} points, {total} stars")


if __name__ == "__main__":
    main()
