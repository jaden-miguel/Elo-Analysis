import pandas as pd
import plotly.express as px

# Load aggregated team points
summary = pd.read_csv("data/team_points.csv")
elo = pd.read_csv("data/team_elo.csv")

# Sort teams alphabetically for consistent bars
summary = summary.sort_values(by=["team", "season"])

fig = px.bar(
    summary,
    x="team",
    y="points",
    color="season",
    barmode="group",
    title="Premier League Points by Team (2022-23 vs 2023-24)",
    labels={"points": "Total Points", "team": "Team"},
    height=600,
)

fig.update_layout(xaxis_tickangle=45)

fig.show()

# Line chart for Elo ratings (top 6 teams by final rating)
final_ratings = elo.sort_values("date").groupby("team")["rating"].last()
top_teams = final_ratings.sort_values(ascending=False).head(6).index
elo_top = elo[elo["team"].isin(top_teams)]

fig2 = px.line(
    elo_top,
    x="date",
    y="rating",
    color="team",
    title="Elo Rating Over Time (Top Teams)",
    height=600,
)

fig2.show()
