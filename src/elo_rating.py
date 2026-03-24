import json
import pandas as pd
from collections import defaultdict

START_RATING = 1500
K_FACTOR = 30


def load_raw_matches(path="data/matches.json") -> pd.DataFrame:
    """Return raw match results with home/away info."""
    with open(path) as f:
        matches = json.load(f)
    rows = []
    for match in matches:
        score = match.get("score", {}).get("ft", [None, None])
        if None in score:
            continue
        rows.append({
            "season": match.get("season"),
            "date": match.get("date"),
            "home_team": match.get("team1"),
            "away_team": match.get("team2"),
            "home_goals": score[0],
            "away_goals": score[1],
        })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")


def calculate_elo(df: pd.DataFrame, k: int = K_FACTOR) -> pd.DataFrame:
    """Compute Elo rating history for each team."""
    ratings = defaultdict(lambda: START_RATING)
    history = []
    for _, row in df.iterrows():
        home, away = row.home_team, row.away_team
        r_home = ratings[home]
        r_away = ratings[away]

        if row.home_goals > row.away_goals:
            res_home, res_away = 1, 0
        elif row.home_goals < row.away_goals:
            res_home, res_away = 0, 1
        else:
            res_home = res_away = 0.5

        exp_home = 1 / (1 + 10 ** ((r_away - r_home) / 400))
        exp_away = 1 / (1 + 10 ** ((r_home - r_away) / 400))

        new_r_home = r_home + k * (res_home - exp_home)
        new_r_away = r_away + k * (res_away - exp_away)

        ratings[home] = new_r_home
        ratings[away] = new_r_away

        history.append({
            "date": row.date,
            "team": home,
            "opponent": away,
            "season": row.season,
            "rating": new_r_home,
        })
        history.append({
            "date": row.date,
            "team": away,
            "opponent": home,
            "season": row.season,
            "rating": new_r_away,
        })

    return pd.DataFrame(history)


if __name__ == "__main__":
    raw = load_raw_matches()
    elo_df = calculate_elo(raw)
    elo_df.to_csv("data/team_elo.csv", index=False)
    print("✅ Saved: data/team_elo.csv")
