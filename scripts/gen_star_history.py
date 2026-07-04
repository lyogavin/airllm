#!/usr/bin/env python3
"""Generate a star-history chart for a GitHub repo, using the authenticated GitHub API.

Unlike the anonymous star-history.com SVG API (which is rate-limited and returns an
empty chart for large repos), this samples stargazer timestamps directly with a token,
so it works reliably for repos with tens of thousands of stars. Output is a PNG, which
always renders in GitHub markdown.

Usage:
    GITHUB_TOKEN=... python scripts/gen_star_history.py [owner/repo] [output.png]
"""
import datetime
import json
import math
import os
import sys
import urllib.request

REPO = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("REPO", "lyogavin/airllm")
OUT = sys.argv[2] if len(sys.argv) > 2 else "assets/star-history.png"
TOKEN = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
PER_PAGE = 100
MAX_SAMPLES = 30


def gh(url, accept="application/vnd.github+json"):
    headers = {"Accept": accept, "User-Agent": "airllm-star-history"}
    if TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


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

    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys, color="#e34a4a", linewidth=2.5, marker="o", markersize=4)
    plt.fill_between(xs, ys, color="#e34a4a", alpha=0.08)
    plt.title(f"Star History — {REPO}", fontsize=15)
    plt.ylabel("GitHub Stars", fontsize=12)
    plt.xlabel("Date", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()

    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    plt.savefig(OUT, dpi=130)
    print(f"wrote {OUT}: {len(points)} points, {total} stars")


if __name__ == "__main__":
    main()
