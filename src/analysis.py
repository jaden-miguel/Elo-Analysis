from data_processing import load_matches, calculate_team_points
from elo_rating import load_raw_matches, calculate_elo


def build_summary():
    df_matches = load_matches()
    summary = calculate_team_points(df_matches)
    summary.to_csv("data/team_points.csv", index=False)
    print("✅ Summary saved to data/team_points.csv")

    raw = load_raw_matches()
    elo_df = calculate_elo(raw)
    elo_df.to_csv("data/team_elo.csv", index=False)
    final_elo = elo_df.sort_values("date").groupby("team")["rating"].last().reset_index()
    final_elo.to_csv("data/final_elo.csv", index=False)
    print("✅ Elo ratings saved to data/team_elo.csv")


if __name__ == "__main__":
    build_summary()
