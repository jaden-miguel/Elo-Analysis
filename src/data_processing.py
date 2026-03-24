import json
import pandas as pd

POINTS = {
    'win': 3,
    'draw': 1,
    'loss': 0,
}


def load_matches(path="data/matches.json") -> pd.DataFrame:
    with open(path) as f:
        matches = json.load(f)
    rows = []
    for match in matches:
        team1 = match.get("team1")
        team2 = match.get("team2")
        score = match.get("score", {}).get("ft", [None, None])
        if None in score:
            continue
        date = match.get("date")
        rows.append({
            "season": match.get("season"),
            "date": date,
            "team": team1,
            "opponent": team2,
            "goals_for": score[0],
            "goals_against": score[1],
            "venue": "home",
        })
        rows.append({
            "season": match.get("season"),
            "date": date,
            "team": team2,
            "opponent": team1,
            "goals_for": score[1],
            "goals_against": score[0],
            "venue": "away",
        })
    return pd.DataFrame(rows)


def calculate_team_points(df: pd.DataFrame) -> pd.DataFrame:
    def result(row):
        if row.goals_for > row.goals_against:
            return 'win'
        elif row.goals_for < row.goals_against:
            return 'loss'
        return 'draw'

    df['result'] = df.apply(result, axis=1)
    df['points'] = df['result'].map(POINTS)
    summary = df.groupby(['team', 'season'])['points'].sum().reset_index()
    return summary


if __name__ == "__main__":
    df_matches = load_matches()
    summary = calculate_team_points(df_matches)
    summary.to_csv("data/team_points.csv", index=False)
    print("✅ Saved: data/team_points.csv")
