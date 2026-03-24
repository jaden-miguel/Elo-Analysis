"""
Premier League Title-Race Prediction Dashboard
===============================================
Generates data/prediction_dashboard.html — a self-contained, dark-themed
page with:
  - Dixon-Coles win-probability bars (pure HTML/CSS — no Plotly for this
    section, so team badges sit perfectly beside each bar at full quality)
  - Last-6 form badges (W / D / L) for every team
  - Current standings table with goal difference, Elo, and title %
  - Interactive Plotly Elo-trajectory chart for the top teams

Run from the project root:
    python src/predict_dashboard.py
"""

import math
import os
import sys
import webbrowser

import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prediction import (
    load_season_data,
    run_monte_carlo,
    compute_elo_history,
    compute_standings,
    compute_form,
    _simple_elo,
)

# ---------------------------------------------------------------------------
# Team identity: colours and badge URLs
# ---------------------------------------------------------------------------

_PL50  = "https://resources.premierleague.com/premierleague/badges/50"
_PL100 = "https://resources.premierleague.com/premierleague/badges/100"

TEAM_COLORS: dict[str, str] = {
    "Arsenal":            "#EF0107",
    "Aston Villa":        "#7B003C",
    "Bournemouth":        "#DA291C",
    "Brentford":          "#E30613",
    "Brighton":           "#0057B8",
    "Chelsea":            "#034694",
    "Crystal Palace":     "#1B458F",
    "Everton":            "#003399",
    "Fulham":             "#CC0000",
    "Ipswich":            "#0044A9",
    "Leicester":          "#003090",
    "Liverpool":          "#C8102E",
    "Man City":           "#6CABDD",
    "Man Utd":            "#DA291C",
    "Newcastle":          "#241F20",
    "Nottm Forest":       "#DD0000",
    "Southampton":        "#D71920",
    "Tottenham":          "#132257",
    "West Ham":           "#7A263A",
    "Wolves":             "#FDB913",
    "Burnley":            "#6C1D45",
    "Leeds":              "#FFCD00",
    "Sheffield Utd":      "#EE2737",
    "Sunderland":         "#EB172B",
    "Luton":              "#F78F1E",
    "Norwich":            "#00A650",
    "Middlesbrough":      "#E01A17",
}

# 100 px badges — one size, no blurriness
TEAM_LOGOS: dict[str, str] = {
    "Arsenal":            f"{_PL100}/t3.png",
    "Aston Villa":        f"{_PL100}/t7.png",
    "Bournemouth":        f"{_PL100}/t91.png",
    "Brentford":          f"{_PL100}/t94.png",
    "Brighton":           f"{_PL100}/t36.png",
    "Chelsea":            f"{_PL100}/t8.png",
    "Crystal Palace":     f"{_PL100}/t31.png",
    "Everton":            f"{_PL100}/t11.png",
    "Fulham":             f"{_PL100}/t54.png",
    "Ipswich":            f"{_PL100}/t40.png",
    "Leicester":          f"{_PL100}/t13.png",
    "Liverpool":          f"{_PL100}/t14.png",
    "Man City":           f"{_PL100}/t43.png",
    "Man Utd":            f"{_PL100}/t1.png",
    "Newcastle":          f"{_PL100}/t4.png",
    "Nottm Forest":       f"{_PL100}/t17.png",
    "Southampton":        f"{_PL100}/t20.png",
    "Tottenham":          f"{_PL100}/t6.png",
    "West Ham":           f"{_PL100}/t21.png",
    "Wolves":             f"{_PL100}/t39.png",
    "Burnley":            f"{_PL100}/t90.png",
    "Leeds":              f"{_PL100}/t2.png",
    "Sheffield Utd":      f"{_PL100}/t49.png",
    "Sunderland":         f"{_PL100}/t56.png",
    "Luton":              f"{_PL100}/t102.png",
}

# Canonical short name for display
_ALIASES: dict[str, str] = {
    "Arsenal FC":                   "Arsenal",
    "AFC Bournemouth":              "Bournemouth",
    "Aston Villa FC":               "Aston Villa",
    "Brentford FC":                 "Brentford",
    "Brighton & Hove Albion FC":    "Brighton",
    "Chelsea FC":                   "Chelsea",
    "Crystal Palace FC":            "Crystal Palace",
    "Everton FC":                   "Everton",
    "Fulham FC":                    "Fulham",
    "Ipswich Town FC":              "Ipswich",
    "Leicester City FC":            "Leicester",
    "Liverpool FC":                 "Liverpool",
    "Manchester City FC":           "Man City",
    "Manchester United FC":         "Man Utd",
    "Newcastle United FC":          "Newcastle",
    "Nottingham Forest FC":         "Nottm Forest",
    "Southampton FC":               "Southampton",
    "Tottenham Hotspur FC":         "Tottenham",
    "West Ham United FC":           "West Ham",
    "Wolverhampton Wanderers FC":   "Wolves",
    "Burnley FC":                   "Burnley",
    "Leeds United FC":              "Leeds",
    "Sheffield United FC":          "Sheffield Utd",
    "Sunderland AFC":               "Sunderland",
    "Luton Town FC":                "Luton",
    # without FC suffix (older seasons)
    "Arsenal":                      "Arsenal",
    "AFC Bournemouth":              "Bournemouth",
    "Aston Villa":                  "Aston Villa",
    "Brentford":                    "Brentford",
    "Brighton & Hove Albion":       "Brighton",
    "Chelsea":                      "Chelsea",
    "Crystal Palace":               "Crystal Palace",
    "Everton":                      "Everton",
    "Fulham":                       "Fulham",
    "Ipswich Town":                 "Ipswich",
    "Leicester City":               "Leicester",
    "Liverpool":                    "Liverpool",
    "Manchester City":              "Man City",
    "Manchester United":            "Man Utd",
    "Newcastle United":             "Newcastle",
    "Nottingham Forest":            "Nottm Forest",
    "Southampton":                  "Southampton",
    "Tottenham Hotspur":            "Tottenham",
    "West Ham United":              "West Ham",
    "Wolverhampton Wanderers":      "Wolves",
    "Burnley":                      "Burnley",
    "Leeds United":                 "Leeds",
    "Sheffield United":             "Sheffield Utd",
    "Sunderland":                   "Sunderland",
    "Luton Town":                   "Luton",
}

DEFAULT_COLOR = "#4A90E2"
DARK_BG  = "#0D0D1E"
CARD_BG  = "#14142B"
PANEL_BG = "#1A1A35"


def sname(raw: str) -> str:
    """Convert raw data team name to canonical short display name."""
    return _ALIASES.get(raw, raw)


def tcolor(raw: str) -> str:
    return TEAM_COLORS.get(sname(raw), DEFAULT_COLOR)


def tlogo(raw: str) -> str:
    return TEAM_LOGOS.get(sname(raw), "")


# ---------------------------------------------------------------------------
# HTML building blocks
# ---------------------------------------------------------------------------

def _form_badges_html(form_str: str, max_show: int = 6) -> str:
    """Generate coloured W/D/L badge spans for a form string."""
    color_map = {"W": "#22c55e", "D": "#f59e0b", "L": "#ef4444"}
    badges = []
    # Show most-recent on the right — form_str is oldest→newest
    visible = form_str[-max_show:] if len(form_str) > max_show else form_str
    for ch in visible:
        c = color_map.get(ch, "#555")
        badges.append(
            f'<span style="display:inline-flex;align-items:center;justify-content:center;'
            f'width:20px;height:20px;border-radius:4px;background:{c};'
            f'font-size:10px;font-weight:800;color:white;flex-shrink:0;">{ch}</span>'
        )
    return "".join(badges)


def _prob_bars_html(win_probs: dict, cur_points: dict, cur_gd: dict,
                    elo_ratings: dict, form_strings: dict) -> str:
    """
    Build the full probability-bar section as pure HTML/CSS.
    Teams are sorted best → worst.  Each row has:
      logo | name | form badges | bar track | percentage
    """
    sorted_teams = sorted(win_probs.items(), key=lambda x: x[1], reverse=True)
    max_prob = max((p for _, p in sorted_teams), default=1e-6)

    rows_html = []
    for rank, (team, prob) in enumerate(sorted_teams):
        display = sname(team)
        logo    = tlogo(team)
        color   = tcolor(team)
        pts     = cur_points.get(team, 0)
        gd      = cur_gd.get(team, 0)
        elo     = elo_ratings.get(team, 1500)
        form    = form_strings.get(team, "")
        pct     = prob * 100

        # Bar fill width relative to the leader (leader = 80 % of track)
        fill_pct = (prob / max_prob) * 80 if max_prob > 0 else 0

        is_leader    = (rank == 0)
        is_contender = (pct >= 0.5)
        is_eliminated = (pct < 0.01)

        row_bg = (
            "rgba(255,215,0,0.06)" if is_leader else
            "rgba(255,255,255,0.02)" if is_contender else
            "transparent"
        )
        row_border = (
            f"border-left:3px solid {color};" if not is_eliminated else
            "border-left:3px solid rgba(255,255,255,0.1);"
        )
        bar_color   = color if not is_eliminated else "rgba(255,255,255,0.12)"
        text_color  = "white" if not is_eliminated else "#555"
        name_weight = "700" if is_leader else "500" if is_contender else "400"

        logo_html = (
            f'<img src="{logo}" width="36" height="36" '
            f'style="object-fit:contain;flex-shrink:0;" '
            f'onerror="this.style.display=\'none\'">'
            if logo else
            f'<div style="width:36px;height:36px;background:{color};'
            f'border-radius:50%;flex-shrink:0;"></div>'
        )

        form_html  = _form_badges_html(form)
        pct_label  = f"{pct:.1f}%" if pct >= 0.1 else (f"{pct:.2f}%" if pct > 0 else "0%")
        elim_badge = (
            '<span style="font-size:10px;color:#555;margin-left:8px;">Elim.</span>'
            if is_eliminated else ""
        )

        rows_html.append(f"""
<div style="display:flex;align-items:center;gap:10px;padding:7px 16px 7px 12px;
            border-radius:8px;background:{row_bg};{row_border}
            margin-bottom:3px;">

  <!-- Badge -->
  {logo_html}

  <!-- Name + form -->
  <div style="min-width:130px;max-width:150px;flex-shrink:0;">
    <div style="font-weight:{name_weight};font-size:0.88rem;color:{text_color};
                white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
      {display}
    </div>
    <div style="display:flex;gap:3px;margin-top:4px;">
      {form_html}
    </div>
  </div>

  <!-- Bar track -->
  <div style="flex:1;height:30px;background:rgba(255,255,255,0.05);
              border-radius:6px;overflow:hidden;position:relative;">
    <div style="height:100%;width:{fill_pct:.2f}%;background:{bar_color};
                border-radius:6px;min-width:{'4px' if not is_eliminated else '0px'};
                box-shadow:{'0 0 12px ' + color + '66' if is_leader else 'none'};
                transition:width 0.8s ease;">
    </div>
  </div>

  <!-- Percentage -->
  <div style="min-width:52px;text-align:right;font-size:0.88rem;
              font-weight:700;color:{color if not is_eliminated else '#555'};">
    {pct_label}{elim_badge}
  </div>

</div>""")

    return "\n".join(rows_html)


def _standings_table_html(teams: list, cur_points: dict, cur_gd: dict,
                           gf: dict, elo_ratings: dict,
                           win_probs: dict, form_strings: dict) -> str:
    """Current-standings table with colour-coded rows."""
    ranked = sorted(teams,
                    key=lambda t: (cur_points.get(t, 0), cur_gd.get(t, 0),
                                   gf.get(t, 0)),
                    reverse=True)
    n = len(ranked)
    rows = []
    for pos, team in enumerate(ranked, 1):
        pts   = cur_points.get(team, 0)
        gd_v  = cur_gd.get(team, 0)
        elo_v = elo_ratings.get(team, 1500)
        prob  = win_probs.get(team, 0) * 100
        form  = form_strings.get(team, "")
        logo  = tlogo(team)
        color = tcolor(team)

        if pos == 1:
            bg = "rgba(255,215,0,0.09)"
        elif pos <= 4:
            bg = "rgba(59,130,246,0.08)"
        elif pos >= n - 2:
            bg = "rgba(239,68,68,0.08)"
        else:
            bg = "transparent"

        logo_html = (
            f'<img src="{logo}" width="22" height="22" '
            f'style="vertical-align:middle;margin-right:6px;object-fit:contain;" '
            f'onerror="this.style.display:none">'
            if logo else ""
        )

        prob_bar = (
            f'<div style="display:inline-flex;align-items:center;gap:6px;">'
            f'<div style="width:{min(prob*1.8, 90):.0f}px;height:6px;'
            f'background:{color};border-radius:3px;flex-shrink:0;"></div>'
            f'<span style="font-size:0.8rem;">{prob:.1f}%</span>'
            f'</div>'
        )

        form_html = _form_badges_html(form, max_show=5)

        rows.append(
            f'<tr style="background:{bg};">'
            f'<td style="text-align:center;color:#777;font-size:0.8rem;">{pos}</td>'
            f'<td>{logo_html}<span style="font-size:0.88rem;">{sname(team)}</span></td>'
            f'<td style="text-align:center;font-weight:700;">{pts}</td>'
            f'<td style="text-align:center;color:#999;font-size:0.85rem;">{gd_v:+d}</td>'
            f'<td style="text-align:center;color:#999;font-size:0.8rem;">{elo_v:.0f}</td>'
            f'<td><div style="display:flex;gap:3px;">{form_html}</div></td>'
            f'<td>{prob_bar}</td>'
            f'</tr>'
        )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Elo chart (Plotly)
# ---------------------------------------------------------------------------

def build_elo_chart(elo_history: pd.DataFrame, elo_ratings: dict) -> str:
    """Return HTML snippet for the Elo trajectory Plotly chart."""
    import plotly.io as pio

    top8 = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)[:8]
    top8_names = [t[0] for t in top8]

    fig = go.Figure()
    for team in top8_names:
        df = elo_history[elo_history["team"] == team].sort_values("date")
        if df.empty:
            continue
        color = tcolor(team)
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["rating"],
            mode="lines",
            name=sname(team),
            line=dict(color=color, width=2.5),
            hovertemplate=(
                f"<b>{sname(team)}</b><br>"
                "Date: %{x|%d %b %Y}<br>"
                "Elo: %{y:.0f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        paper_bgcolor=PANEL_BG,
        plot_bgcolor=PANEL_BG,
        font=dict(color="white", family="Segoe UI, Arial"),
        height=380,
        margin=dict(l=60, r=20, t=40, b=50),
        legend=dict(font=dict(color="white", size=10),
                    bgcolor="rgba(20,20,40,0.8)",
                    bordercolor="rgba(255,255,255,0.12)",
                    borderwidth=1),
        xaxis=dict(gridcolor="rgba(255,255,255,0.07)",
                   tickfont=dict(color="#aaa")),
        yaxis=dict(title="Elo Rating",
                   title_font=dict(color="#aaa", size=11),
                   gridcolor="rgba(255,255,255,0.07)",
                   tickfont=dict(color="#aaa")),
    )

    return pio.to_html(fig, full_html=False, include_plotlyjs="cdn",
                       config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# Full HTML assembly
# ---------------------------------------------------------------------------

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: #0D0D1E;
  color: #f0f0f0;
  font-family: 'Segoe UI', Arial, sans-serif;
  min-height: 100vh;
}
.site-header {
  background: linear-gradient(135deg, #180030 0%, #3d006a 50%, #180030 100%);
  border-bottom: 3px solid #7c3aed;
  padding: 22px 36px;
  display: flex; align-items: center; gap: 18px;
}
.pl-badge {
  width: 50px; height: 50px;
  background: #7c3aed; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 20px; font-weight: 900; color: white; flex-shrink: 0;
  box-shadow: 0 0 22px rgba(124,58,237,0.7);
  letter-spacing: -1px;
}
.header-text h1 {
  font-size: 1.8rem; font-weight: 900;
  background: linear-gradient(90deg, #fff, #c084fc);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.header-text p { color: #a78bfa; font-size: 0.88rem; margin-top: 4px; }

.main { padding: 24px 32px; }

/* Winner spotlight */
.winner-card {
  border-radius: 14px;
  padding: 26px 30px;
  display: flex; align-items: center; gap: 26px;
  margin-bottom: 24px;
}
.winner-info h2 {
  font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1.8px;
  color: #aaa; margin-bottom: 6px;
}
.winner-name { font-size: 2.6rem; font-weight: 900; line-height: 1.1; }
.winner-pct { font-size: 1.1rem; color: #ccc; margin-top: 6px; }
.runners { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 12px; }
.runner {
  background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1);
  border-radius: 8px; padding: 7px 12px;
  display: flex; align-items: center; gap: 8px; font-size: 0.87rem;
}
.runner-pct { color: #888; font-size: 0.78rem; }

/* Cards */
.card {
  background: #14142B;
  border: 1px solid rgba(255,255,255,0.07);
  border-radius: 12px; overflow: hidden;
  margin-bottom: 24px;
}
.card-header {
  padding: 13px 18px;
  font-size: 0.73rem; font-weight: 700; letter-spacing: 1.4px;
  text-transform: uppercase; color: #6666aa;
  border-bottom: 1px solid rgba(255,255,255,0.06);
}
.card-body { padding: 16px; }

/* Two-column grid */
.two-col {
  display: grid; grid-template-columns: 1fr 1fr; gap: 22px;
  margin-bottom: 24px;
}
@media (max-width: 960px) { .two-col { grid-template-columns: 1fr; } }

/* Standings table */
.standings-tbl { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
.standings-tbl th {
  padding: 8px 10px; text-align: left;
  font-size: 0.7rem; letter-spacing: 1px; text-transform: uppercase;
  color: #555577; border-bottom: 1px solid rgba(255,255,255,0.07);
}
.standings-tbl td {
  padding: 6px 10px; border-bottom: 1px solid rgba(255,255,255,0.03);
  vertical-align: middle;
}
.standings-tbl tr:hover td { background: rgba(255,255,255,0.03); }

/* Legend */
.legend {
  display: flex; gap: 16px; padding: 10px 18px;
  font-size: 0.72rem; color: #666;
  border-top: 1px solid rgba(255,255,255,0.05);
}
.pill {
  width: 10px; height: 10px; border-radius: 2px;
  display: inline-block; margin-right: 5px; vertical-align: middle;
}

/* Model callout */
.model-callout {
  background: rgba(124,58,237,0.08);
  border: 1px solid rgba(124,58,237,0.25);
  border-radius: 10px; padding: 14px 20px;
  font-size: 0.82rem; color: #c4b5fd; line-height: 1.6;
  margin-bottom: 24px;
}
.model-callout strong { color: #ddd; }
"""


def build_html(
    win_probs: dict,
    cur_points: dict,
    cur_gd: dict,
    gf: dict,
    elo_ratings: dict,
    form_scores: dict,
    form_strings: dict,
    teams: list,
    target_season: str,
    remaining_count: int,
    is_complete: bool,
    elo_chart_html: str,
) -> str:
    winner      = max(win_probs, key=win_probs.get)
    winner_prob = win_probs[winner] * 100
    w_color     = tcolor(winner)
    w_logo      = tlogo(winner)
    w_short     = sname(winner)
    season_lbl  = target_season.replace("-", "/")
    match_lbl   = "Season complete" if is_complete else f"{remaining_count} fixtures remaining"

    logo_img = (
        f'<img src="{w_logo}" width="100" height="100" '
        f'style="filter:drop-shadow(0 0 18px {w_color});object-fit:contain;" '
        f'onerror="this.style.display=\'none\'">'
        if w_logo else ""
    )

    # Runners-up (2nd–4th)
    by_prob = sorted(win_probs.items(), key=lambda x: x[1], reverse=True)
    runners_html = ""
    for t, p in by_prob[1:4]:
        c  = tcolor(t)
        lg = tlogo(t)
        lgi = (f'<img src="{lg}" width="28" height="28" style="object-fit:contain;" '
               f'onerror="this.style.display:none">' if lg else "")
        runners_html += (
            f'<div class="runner">{lgi}'
            f'<span style="color:{c};font-weight:700;">{sname(t)}</span>'
            f'<span class="runner-pct">{p*100:.1f}%</span></div>'
        )

    prob_bars = _prob_bars_html(win_probs, cur_points, cur_gd, elo_ratings, form_strings)
    standings = _standings_table_html(teams, cur_points, cur_gd, gf, elo_ratings,
                                       win_probs, form_strings)

    model_explanation = (
        "<strong>Model:</strong> Dixon-Coles time-weighted Poisson regression — "
        "estimates separate attack and defensive strength for every club using "
        "all historical results (exponential time decay so recent games count more), "
        "then simulates each remaining fixture by sampling goals from a Poisson "
        "distribution and adjusts for current-season form (last 6 games). "
        "10,000 Monte Carlo runs produce the title-win probabilities shown above."
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Premier League {season_lbl} — Title Prediction</title>
<style>{CSS}</style>
</head>
<body>

<header class="site-header">
  <div class="pl-badge">PL</div>
  <div class="header-text">
    <h1>Premier League {season_lbl} — Title Race</h1>
    <p>Dixon-Coles Poisson Model &nbsp;·&nbsp; {match_lbl} &nbsp;·&nbsp; 10,000 Monte Carlo simulations</p>
  </div>
</header>

<div class="main">

  <!-- Model explanation -->
  <div class="model-callout">{model_explanation}</div>

  <!-- Winner spotlight -->
  <div class="winner-card" style="
    background:linear-gradient(135deg, {CARD_BG} 0%,
      rgba({_hex_to_rgb(w_color)},0.18) 100%);
    border:1px solid rgba({_hex_to_rgb(w_color)},0.4);
    box-shadow:0 0 50px rgba({_hex_to_rgb(w_color)},0.15);
  ">
    <div style="flex-shrink:0;">{logo_img}</div>
    <div class="winner-info">
      <h2>Predicted Champion</h2>
      <div class="winner-name" style="color:{w_color};
        text-shadow:0 0 40px rgba({_hex_to_rgb(w_color)},0.5);">
        {w_short}
      </div>
      <div class="winner-pct">
        <span style="font-size:1.6rem;font-weight:900;color:{w_color};">
          {winner_prob:.1f}%
        </span> title probability
      </div>
      <div class="runners">
        <span style="color:#666;font-size:0.82rem;align-self:center;">Also in contention:</span>
        {runners_html}
      </div>
    </div>
  </div>

  <!-- Two-column: probability chart + standings -->
  <div class="two-col">

    <!-- Probability bars (pure HTML/CSS — perfect badge alignment) -->
    <div class="card">
      <div class="card-header">Title Win Probability</div>
      <div class="card-body" style="padding:12px 8px;">
        {prob_bars}
      </div>
    </div>

    <!-- Standings -->
    <div class="card">
      <div class="card-header">Current Standings</div>
      <div style="overflow-y:auto;max-height:640px;">
        <table class="standings-tbl">
          <thead>
            <tr>
              <th>#</th>
              <th>Club</th>
              <th>Pts</th>
              <th>GD</th>
              <th>Elo</th>
              <th>Form</th>
              <th>Title %</th>
            </tr>
          </thead>
          <tbody>{standings}</tbody>
        </table>
      </div>
      <div class="legend">
        <span><span class="pill" style="background:rgba(255,215,0,0.5)"></span>Leader</span>
        <span><span class="pill" style="background:rgba(59,130,246,0.5)"></span>UCL spots (Top 4)</span>
        <span><span class="pill" style="background:rgba(239,68,68,0.4)"></span>Relegation zone</span>
      </div>
    </div>

  </div>

  <!-- Elo history chart -->
  <div class="card">
    <div class="card-header">Elo Rating Trajectory — Top 8 Teams (all seasons)</div>
    <div style="padding:4px;">
      {elo_chart_html}
    </div>
  </div>

</div>
</body>
</html>"""


def _hex_to_rgb(hex_color: str) -> str:
    """Convert '#RRGGBB' to 'r,g,b' string for use in rgba()."""
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = h[0]*2 + h[1]*2 + h[2]*2
    try:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"{r},{g},{b}"
    except Exception:
        return "74,144,226"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs("data", exist_ok=True)
    data_path = "data/matches.json"

    if not os.path.exists(data_path):
        print("data/matches.json not found — fetching data first...")
        import subprocess
        subprocess.run(
            ["py", os.path.join(os.path.dirname(__file__), "data_fetch.py")],
            check=True,
        )

    print("Loading match data...")
    completed, upcoming = load_season_data(data_path)

    if completed.empty:
        print("ERROR: No completed matches found.")
        return

    seasons_with_data = (
        completed.groupby("season").size().sort_index().index.tolist()
    )
    upcoming_seasons = (
        upcoming["season"].unique().tolist() if not upcoming.empty else []
    )
    all_seasons  = sorted(set(seasons_with_data + upcoming_seasons))
    target_season = all_seasons[-1]
    if target_season not in seasons_with_data:
        target_season = seasons_with_data[-1]

    s_done = completed[completed["season"] == target_season]
    s_todo = (upcoming[upcoming["season"] == target_season]
              if not upcoming.empty else pd.DataFrame())

    remaining_count = len(s_todo)
    is_complete     = (remaining_count == 0)

    print(f"Target season : {target_season}")
    print(f"Completed     : {len(s_done)} matches")
    print(f"Remaining     : {remaining_count} fixtures")

    # Compute form before Monte Carlo (reused for display)
    print("Computing form...")
    form_scores, form_strings = compute_form(completed, target_season)

    print("Running Monte Carlo (10,000 iterations)...")
    win_probs, cur_points, cur_gd, elo_ratings, teams = run_monte_carlo(
        completed, upcoming, target_season, n_simulations=10_000
    )

    # Current season GF for tiebreakers
    _, _, gf = compute_standings(completed, target_season)

    # Elo history for the chart
    elo_history = compute_elo_history(completed)

    # Console summary
    print("\nTitle-race probabilities:")
    for team, prob in sorted(win_probs.items(), key=lambda x: x[1], reverse=True):
        bar = "#" * int(prob * 40)
        print(f"  {sname(team):<22} {prob*100:6.2f}%  {bar}")

    print("\nBuilding Elo chart...")
    elo_chart_html = build_elo_chart(elo_history, elo_ratings)

    print("Assembling dashboard...")
    html = build_html(
        win_probs, cur_points, cur_gd, gf, elo_ratings,
        form_scores, form_strings,
        teams, target_season, remaining_count, is_complete,
        elo_chart_html,
    )

    out_path = "data/prediction_dashboard.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    abs_path = os.path.abspath(out_path).replace(os.sep, "/")
    print(f"\nDashboard saved: {abs_path}")
    print("Opening in browser...")
    webbrowser.open(f"file:///{abs_path}")


if __name__ == "__main__":
    main()
