"""Refresh the two-repository star snapshot and README badge (stdlib only)."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

REPOSITORIES = ("SIAT-INVS/CarlaFLCAV", "KevinLADLee/carla_dataset_tools")
ASSETS = Path(__file__).resolve().parents[1] / "docs" / "assets"


def fetch_counts():
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "CarlaFLCAV-star-counter"}
    if os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["GITHUB_TOKEN"]
    counts = {}
    for repo in REPOSITORIES:
        request = Request("https://api.github.com/repos/" + repo, headers=headers)
        with urlopen(request, timeout=30) as response:
            data = json.load(response)
        count = data.get("stargazers_count")
        if data.get("full_name", "").lower() != repo.lower() or type(count) is not int or count < 0:
            raise ValueError("Invalid GitHub star response for " + repo)
        counts[repo] = count
    return counts


def save_counts(counts, checked_at, source="GitHub REST API", directory=ASSETS):
    if set(counts) != set(REPOSITORIES) or any(type(n) is not int or n < 0 for n in counts.values()):
        raise ValueError("Both repository counts must be nonnegative integers")
    total = sum(counts.values())
    snapshot = {"repositories": counts, "total": total, "checked_at": checked_at, "source": source}
    value = f"{total:,}"
    # Match Shields.io's for-the-badge font, text spacing, and 28px height.
    label_width = 70.5
    value_text_width = sum(8.5 if char.isdigit() else 5 for char in value) + 14.75
    value_width = value_text_width + 24
    width = label_width + value_width
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="28" role="img" aria-label="GitHub stars: {total}">
<title>GitHub stars: {total}</title>
<g shape-rendering="crispEdges">
<rect width="{label_width}" height="28" fill="#555"/>
<rect x="{label_width}" width="{value_width}" height="28" fill="#3c5488"/>
</g>
<g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" text-rendering="geometricPrecision" font-size="100">
<text transform="scale(.1)" x="{label_width * 5}" y="175" textLength="465">GITHUB</text>
<text transform="scale(.1)" x="{(label_width + value_width / 2) * 10}" y="175" textLength="{value_text_width * 10}" font-weight="bold">★ {value}</text>
</g>
</svg>\n'''
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in (("stars.json", json.dumps(snapshot, indent=2) + "\n"), ("combined-stars.svg", svg)):
        temporary = directory / (name + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(directory / name)
    return total


def main():
    # Fetch and validate both repositories before writing either output.
    counts = fetch_counts()
    total = save_counts(counts, datetime.now(timezone.utc).isoformat(timespec="seconds"))
    print(f"Updated combined stars: {total}")


if __name__ == "__main__":
    main()
