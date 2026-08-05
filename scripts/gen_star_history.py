#!/usr/bin/env python3
"""Generate a star-history chart for a GitHub repo, using the authenticated GitHub API.

Unlike the anonymous star-history.com SVG API (which is rate-limited and returns an
empty chart for large repos), this samples stargazer timestamps directly with a token,
so it works reliably for repos with tens of thousands of stars. Output is a PNG, which
always renders in GitHub markdown.

Usage:
    GITHUB_TOKEN=... python scripts/gen_star_history.py [owner/repo] [output.png theme]...

theme is "light" (default) or "dark". Pass several output/theme pairs to render them all
from a single pass over the API.
"""
import datetime
import json
import os
import sys
import urllib.error
import urllib.request

GRAPHQL_URL = "https://api.github.com/graphql"

REPO = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("REPO", "lyogavin/airllm")

# Remaining args are output/theme pairs. Walking the stargazer connection is by far the
# slowest part of this script, so rendering every theme from one walk beats invoking the
# script once per theme.
_rest = sys.argv[2:]
if _rest:
    TARGETS = [(_rest[i], _rest[i + 1].lower() if i + 1 < len(_rest) else "light")
               for i in range(0, len(_rest), 2)]
else:
    TARGETS = [("assets/star-history.png", os.environ.get("THEME", "light").lower())]

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
        if e.code == 401:
            hint = (
                "\nThe token was rejected outright, which means it is expired or malformed "
                "rather than under-permissioned. Issue a new one and re-run: "
                "gh secret set STAR_HISTORY_TOKEN"
            )
        elif e.code == 403 and "personal access token" in msg.lower():
            # A fine-grained PAT reaching an endpoint its permissions do not cover. This is
            # distinct from a missing token, and reads identically in logs unless called out:
            # fine-grained grants have never been enough for /stargazers.
            hint = (
                "\nA fine-grained PAT cannot read /stargazers. Use a *classic* token with the "
                "public_repo scope (https://github.com/settings/tokens) and re-run: "
                "gh secret set STAR_HISTORY_TOKEN"
            )
        elif e.code == 403 and "stargazers" in url:
            hint = (
                "\nGitHub restricts /stargazers to admins/collaborators. The token in "
                "STAR_HISTORY_TOKEN must belong to one, and must be a classic token with "
                "the public_repo scope."
            )
        raise SystemExit(f"GitHub API {e.code} for {url}: {msg}{hint}") from e


def gql(query, variables):
    payload = json.dumps({"query": query, "variables": variables}).encode()
    headers = {
        "Accept": "application/vnd.github+json",
        "Content-Type": "application/json",
        "User-Agent": "airllm-star-history",
    }
    if TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"
    req = urllib.request.Request(GRAPHQL_URL, data=payload, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = json.load(r)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        extra = "" if TOKEN else "\nNo token found in GITHUB_TOKEN or GH_TOKEN."
        raise SystemExit(f"GitHub GraphQL {e.code}: {detail}{extra}") from e
    if body.get("errors"):
        raise SystemExit(f"GitHub GraphQL error: {json.dumps(body['errors'])}")
    return body["data"]


STARGAZER_QUERY = """
query($owner:String!, $name:String!, $after:String) {
  repository(owner:$owner, name:$name) {
    stargazerCount
    stargazers(first:%d, after:$after, orderBy:{field:STARRED_AT, direction:ASC}) {
      pageInfo { hasNextPage endCursor }
      edges { starredAt }
    }
  }
}
""" % PER_PAGE


def fetch_starred_at():
    """Every stargazer timestamp, oldest first, plus the repo's current star count.

    Uses GraphQL rather than REST. GitHub restricted REST /stargazers to admins and
    collaborators in 2026, and then tightened it again so that fine-grained tokens are
    refused outright -- the same chart broke twice on token permissions. GraphQL serves
    the identical public data without that gate.

    Its cursors encode a timestamp and user id rather than an offset, so unlike REST we
    cannot jump to sampled pages and have to walk the whole connection. That is a few
    hundred requests for a repo this size, which is fine for a once-a-day job.
    """
    owner, _, name = REPO.partition("/")
    stamps, cursor, total = [], None, 0
    while True:
        data = gql(STARGAZER_QUERY, {"owner": owner, "name": name, "after": cursor})
        repo = data.get("repository")
        if repo is None:
            raise SystemExit(f"repo {REPO} not found, or the token cannot see it")
        total = int(repo["stargazerCount"])
        conn = repo["stargazers"]
        stamps.extend(e["starredAt"] for e in conn["edges"])
        if not conn["pageInfo"]["hasNextPage"]:
            return stamps, total
        cursor = conn["pageInfo"]["endCursor"]


def main():
    stamps, total = fetch_starred_at()
    if total <= 0 or not stamps:
        raise SystemExit("repo has no stars")

    # Thin the full history down to a readable number of vertices. Walking every
    # stargazer would plot tens of thousands of points on top of each other.
    step = max(1, len(stamps) // MAX_SAMPLES)
    points = []
    for i in range(0, len(stamps), step):
        dt = datetime.datetime.fromisoformat(stamps[i].replace("Z", "+00:00"))
        points.append((dt, i + 1))

    points.append((datetime.datetime.now(datetime.timezone.utc), total))
    points = sorted(set(points))
    if len(points) < 2:
        raise SystemExit("not enough data points to plot")

    for out, theme in TARGETS:
        render(points, total, out, theme)


def render(points, total, out, theme):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    xs = [d for d, _ in points]
    ys = [c for _, c in points]

    c = THEMES.get(theme, THEMES["light"])
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

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=130, facecolor=c["bg"])
    plt.close(fig)
    print(f"wrote {out} ({theme}): {len(points)} points, {total} stars")


if __name__ == "__main__":
    main()
