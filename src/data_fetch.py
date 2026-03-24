import os
import json
import requests

SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]
BASE_URL = "https://raw.githubusercontent.com/openfootball/football.json/master/{season}/en.1.json"

os.makedirs("data", exist_ok=True)


def extract_matches(data: dict) -> list:
    """Extract matches from both flat and rounds-based JSON formats."""
    if "matches" in data and isinstance(data["matches"], list):
        return data["matches"]
    if "rounds" in data and isinstance(data["rounds"], list):
        matches = []
        for round_data in data["rounds"]:
            matches.extend(round_data.get("matches", []))
        return matches
    return []


def fetch_season(season: str) -> list:
    """Download match data for a given Premier League season."""
    url = BASE_URL.format(season=season)
    print(f"\n  Fetching {season}...")
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        matches = extract_matches(data)
        scored = sum(1 for m in matches if m.get("score", {}) and m["score"].get("ft"))
        print(f"     {len(matches)} matches found ({scored} completed)")
        return matches
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"     Season not available yet — skipping")
        else:
            print(f"     HTTP error: {e}")
        return []
    except Exception as e:
        print(f"     Error: {e}")
        return []


print("Premier League Data Fetch")
print("=" * 35)

all_matches = []
for season in SEASONS:
    matches = fetch_season(season)
    for match in matches:
        match["season"] = season
    all_matches.extend(matches)

with open("data/matches.json", "w") as f:
    json.dump(all_matches, f, indent=2)

print(f"\nSaved: data/matches.json  ({len(all_matches)} total matches)")
