"""
Premier League Prediction + Betting Guide Dashboard
====================================================
Generates data/prediction_dashboard.html — a self-contained dark-themed page.

Sections:
  1. Model explanation callout
  2. Predicted champion spotlight
  3. Two-col: probability bars (HTML/CSS, 100 px badges) + current standings
  4. BEST BETS panel — season markets ranked by edge + confidence
  5. Upcoming fixture breakdown — per-match W/D/L, xG, Over 2.5, BTTS
  6. Elo trajectory chart (Plotly)

Run from the project root:
    python src/predict_dashboard.py
"""

import math
import os
import sys
import webbrowser
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prediction import (
    load_season_data,
    run_monte_carlo,
    compute_elo_history,
    compute_standings,
    compute_goals_form,
    predict_upcoming_fixtures,
    _simple_elo,
)

# ---------------------------------------------------------------------------
# Team identity
# ---------------------------------------------------------------------------

_PL100 = "https://resources.premierleague.com/premierleague/badges/100"

TEAM_COLORS: dict[str, str] = {
    "Arsenal":       "#EF0107", "Aston Villa":   "#7B003C",
    "Bournemouth":   "#DA291C", "Brentford":     "#E30613",
    "Brighton":      "#0057B8", "Chelsea":       "#034694",
    "Crystal Palace":"#1B458F", "Everton":       "#003399",
    "Fulham":        "#CC0000", "Ipswich":       "#0044A9",
    "Leicester":     "#003090", "Liverpool":     "#C8102E",
    "Man City":      "#6CABDD", "Man Utd":       "#DA291C",
    "Newcastle":     "#241F20", "Nottm Forest":  "#DD0000",
    "Southampton":   "#D71920", "Tottenham":     "#132257",
    "West Ham":      "#7A263A", "Wolves":        "#FDB913",
    "Burnley":       "#6C1D45", "Leeds":         "#FFCD00",
    "Sheffield Utd": "#EE2737", "Sunderland":    "#EB172B",
    "Luton":         "#F78F1E", "Norwich":       "#00A650",
}

TEAM_LOGOS: dict[str, str] = {
    "Arsenal":       f"{_PL100}/t3.png",   "Aston Villa":   f"{_PL100}/t7.png",
    "Bournemouth":   f"{_PL100}/t91.png",  "Brentford":     f"{_PL100}/t94.png",
    "Brighton":      f"{_PL100}/t36.png",  "Chelsea":       f"{_PL100}/t8.png",
    "Crystal Palace":f"{_PL100}/t31.png",  "Everton":       f"{_PL100}/t11.png",
    "Fulham":        f"{_PL100}/t54.png",  "Ipswich":       f"{_PL100}/t40.png",
    "Leicester":     f"{_PL100}/t13.png",  "Liverpool":     f"{_PL100}/t14.png",
    "Man City":      f"{_PL100}/t43.png",  "Man Utd":       f"{_PL100}/t1.png",
    "Newcastle":     f"{_PL100}/t4.png",   "Nottm Forest":  f"{_PL100}/t17.png",
    "Southampton":   f"{_PL100}/t20.png",  "Tottenham":     f"{_PL100}/t6.png",
    "West Ham":      f"{_PL100}/t21.png",  "Wolves":        f"{_PL100}/t39.png",
    "Burnley":       f"{_PL100}/t90.png",  "Leeds":         f"{_PL100}/t2.png",
    "Sheffield Utd": f"{_PL100}/t49.png",  "Sunderland":    f"{_PL100}/t56.png",
    "Luton":         f"{_PL100}/t102.png",
}

_ALIASES: dict[str, str] = {
    "Arsenal FC":                 "Arsenal",
    "AFC Bournemouth":            "Bournemouth",
    "Aston Villa FC":             "Aston Villa",
    "Brentford FC":               "Brentford",
    "Brighton & Hove Albion FC":  "Brighton",
    "Chelsea FC":                 "Chelsea",
    "Crystal Palace FC":          "Crystal Palace",
    "Everton FC":                 "Everton",
    "Fulham FC":                  "Fulham",
    "Ipswich Town FC":            "Ipswich",
    "Leicester City FC":          "Leicester",
    "Liverpool FC":               "Liverpool",
    "Manchester City FC":         "Man City",
    "Manchester United FC":       "Man Utd",
    "Newcastle United FC":        "Newcastle",
    "Nottingham Forest FC":       "Nottm Forest",
    "Southampton FC":             "Southampton",
    "Tottenham Hotspur FC":       "Tottenham",
    "West Ham United FC":         "West Ham",
    "Wolverhampton Wanderers FC": "Wolves",
    "Burnley FC":                 "Burnley",
    "Leeds United FC":            "Leeds",
    "Sheffield United FC":        "Sheffield Utd",
    "Sunderland AFC":             "Sunderland",
    "Luton Town FC":              "Luton",
    # names without FC (older seasons)
    "Brighton & Hove Albion":     "Brighton",
    "Wolverhampton Wanderers":    "Wolves",
    "Tottenham Hotspur":          "Tottenham",
    "Manchester City":            "Man City",
    "Manchester United":          "Man Utd",
    "Newcastle United":           "Newcastle",
    "Nottingham Forest":          "Nottm Forest",
    "West Ham United":            "West Ham",
    "Leicester City":             "Leicester",
    "Ipswich Town":               "Ipswich",
    "Crystal Palace":             "Crystal Palace",
    "Sheffield United":           "Sheffield Utd",
    "Luton Town":                 "Luton",
    "Leeds United":               "Leeds",
}

DEFAULT_COLOR = "#4A90E2"
DARK_BG  = "#0D0D1E"
CARD_BG  = "#14142B"
PANEL_BG = "#1A1A35"


def sname(raw: str) -> str:
    return _ALIASES.get(raw, raw)

def tcolor(raw: str) -> str:
    return TEAM_COLORS.get(sname(raw), DEFAULT_COLOR)

def tlogo(raw: str) -> str:
    return TEAM_LOGOS.get(sname(raw), "")

def odds(p: float) -> str:
    """Convert probability to decimal odds string."""
    if p <= 0: return "N/A"
    return f"{1/p:.2f}"

def confidence(p: float) -> tuple[str, str]:
    """Return (label, css-color) confidence tier."""
    if p >= 0.75: return "STRONG",  "#22c55e"
    if p >= 0.60: return "SOLID",   "#84cc16"
    if p >= 0.50: return "MODERATE","#f59e0b"
    return "WEAK", "#ef4444"

def _hex_rgb(h: str) -> str:
    h = h.lstrip("#")
    if len(h) == 3: h = h[0]*2+h[1]*2+h[2]*2
    try:
        return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"
    except Exception:
        return "74,144,226"


# ---------------------------------------------------------------------------
# HTML components
# ---------------------------------------------------------------------------

def _form_badges(form_str: str, n: int = 6) -> str:
    cm = {"W": "#22c55e", "D": "#f59e0b", "L": "#ef4444"}
    out = []
    for ch in form_str[-n:]:
        c = cm.get(ch, "#555")
        out.append(
            f'<span style="display:inline-flex;align-items:center;justify-content:center;'
            f'width:20px;height:20px;border-radius:4px;background:{c};'
            f'font-size:10px;font-weight:800;color:white;flex-shrink:0;">{ch}</span>'
        )
    return "".join(out)


def _logo_img(team: str, size: int = 36) -> str:
    url = tlogo(team)
    if url:
        return (f'<img src="{url}" width="{size}" height="{size}" '
                f'style="object-fit:contain;flex-shrink:0;" '
                f'onerror="this.style.display=\'none\'">')
    c = tcolor(team)
    return (f'<div style="width:{size}px;height:{size}px;background:{c};'
            f'border-radius:50%;flex-shrink:0;"></div>')


# ── Probability bars ─────────────────────────────────────────────────────────

def _prob_bars_html(win_probs, cur_points, cur_gd, elo_ratings, form_strings) -> str:
    by_prob = sorted(win_probs.items(), key=lambda x: x[1], reverse=True)
    max_p   = max((p for _, p in by_prob), default=1e-9)
    rows = []
    for rank, (team, prob) in enumerate(by_prob):
        display  = sname(team)
        color    = tcolor(team)
        pts      = cur_points.get(team, 0)
        elo_v    = elo_ratings.get(team, 1500)
        form     = form_strings.get(team, "")
        pct      = prob * 100
        fill     = (prob / max_p) * 80
        is_lead  = (rank == 0)
        is_cont  = (pct >= 0.5)
        is_elim  = (pct < 0.01)
        row_bg   = ("rgba(255,215,0,0.07)" if is_lead else
                    "rgba(255,255,255,0.02)" if is_cont else "transparent")
        border   = f"border-left:3px solid {color};" if not is_elim else "border-left:3px solid rgba(255,255,255,0.08);"
        bar_col  = color if not is_elim else "rgba(255,255,255,0.1)"
        txt_col  = "white" if not is_elim else "#555"
        wt       = "700" if is_lead else "500" if is_cont else "400"
        pct_lbl  = f"{pct:.1f}%" if pct >= 0.1 else (f"{pct:.2f}%" if pct > 0 else "0%")
        rows.append(f"""
<div style="display:flex;align-items:center;gap:10px;padding:6px 14px 6px 10px;
            border-radius:8px;background:{row_bg};{border}margin-bottom:3px;">
  {_logo_img(team, 36)}
  <div style="min-width:128px;max-width:140px;flex-shrink:0;">
    <div style="font-weight:{wt};font-size:0.87rem;color:{txt_col};
                white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{display}</div>
    <div style="display:flex;gap:3px;margin-top:3px;">{_form_badges(form)}</div>
  </div>
  <div style="flex:1;height:30px;background:rgba(255,255,255,0.05);border-radius:6px;overflow:hidden;">
    <div style="height:100%;width:{fill:.1f}%;background:{bar_col};border-radius:6px;
                min-width:{'3px' if not is_elim else '0'};
                box-shadow:{'0 0 14px '+color+'66' if is_lead else 'none'};"></div>
  </div>
  <div style="min-width:50px;text-align:right;font-size:0.88rem;font-weight:700;
              color:{color if not is_elim else '#555'};">{pct_lbl}</div>
</div>""")
    return "\n".join(rows)


# ── Standings table ───────────────────────────────────────────────────────────

def _standings_table(teams, cur_points, cur_gd, gf, elo_ratings, win_probs, form_strings) -> str:
    ranked = sorted(teams,
                    key=lambda t: (cur_points.get(t,0), cur_gd.get(t,0), gf.get(t,0)),
                    reverse=True)
    n  = len(ranked)
    rows = []
    for pos, team in enumerate(ranked, 1):
        pts  = cur_points.get(team, 0)
        gd   = cur_gd.get(team, 0)
        elo  = elo_ratings.get(team, 1500)
        prob = win_probs.get(team, 0) * 100
        form = form_strings.get(team, "")
        c    = tcolor(team)
        bg   = ("rgba(255,215,0,0.09)" if pos==1 else
                "rgba(59,130,246,0.08)" if pos<=4 else
                "rgba(239,68,68,0.08)" if pos>=n-2 else "transparent")
        prob_bar = (
            f'<div style="display:inline-flex;align-items:center;gap:5px;">'
            f'<div style="width:{min(prob*1.8,90):.0f}px;height:6px;background:{c};border-radius:3px;flex-shrink:0;"></div>'
            f'<span style="font-size:0.78rem;">{prob:.1f}%</span></div>'
        )
        rows.append(
            f'<tr style="background:{bg};">'
            f'<td style="text-align:center;color:#777;font-size:0.78rem;">{pos}</td>'
            f'<td>{_logo_img(team,22)}'
            f'<span style="font-size:0.86rem;margin-left:5px;">{sname(team)}</span></td>'
            f'<td style="text-align:center;font-weight:700;">{pts}</td>'
            f'<td style="text-align:center;color:#999;font-size:0.82rem;">{gd:+d}</td>'
            f'<td style="text-align:center;color:#999;font-size:0.78rem;">{elo:.0f}</td>'
            f'<td><div style="display:flex;gap:2px;">{_form_badges(form,5)}</div></td>'
            f'<td>{prob_bar}</td></tr>'
        )
    return "\n".join(rows)


# ── Season markets / best bets ────────────────────────────────────────────────

def _season_market_rows(label: str, probs: dict, top_n: int = 5,
                         min_p: float = 0.0) -> str:
    by_p = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    rows = []
    for team, p in by_p[:top_n]:
        if p < min_p:
            continue
        c      = tcolor(team)
        lbl, lc = confidence(p)
        rows.append(
            f'<tr>'
            f'<td>{_logo_img(team,24)}'
            f'<span style="font-size:0.86rem;margin-left:6px;">{sname(team)}</span></td>'
            f'<td style="text-align:center;font-weight:700;font-size:1.05rem;color:{c};">{p*100:.1f}%</td>'
            f'<td style="text-align:center;color:#aaa;">{odds(p)}</td>'
            f'<td><span style="background:{lc};color:#000;font-size:0.7rem;font-weight:800;'
            f'padding:2px 7px;border-radius:4px;">{lbl}</span></td>'
            f'</tr>'
        )
    return "\n".join(rows) if rows else "<tr><td colspan='4' style='color:#555;padding:10px;'>—</td></tr>"


def _best_bets_cards(win_probs, top4_probs, rel_probs,
                     upcoming_fixtures: list) -> str:
    """
    Build the ranked 'best bets' section.
    Each bet shows: market, selection, model probability, fair odds,
    confidence tier, and a one-line rationale.
    """
    bets = []

    # Season markets
    winner_team = max(win_probs, key=win_probs.get)
    winner_p    = win_probs[winner_team]
    bets.append({
        "market":    "Title Winner",
        "selection": sname(winner_team),
        "team":      winner_team,
        "prob":      winner_p,
        "why":       f"Current league leader with dominant points gap. "
                     f"Only {100-winner_p*100:.0f}% chance of being caught.",
    })

    # Top-4 certainties
    for team, p in sorted(top4_probs.items(), key=lambda x: x[1], reverse=True)[:3]:
        if p >= 0.70:
            bets.append({
                "market":    "Top 4 Finish",
                "selection": sname(team),
                "team":      team,
                "prob":      p,
                "why":       f"Model gives {p*100:.0f}% chance of Champions League qualification.",
            })

    # Relegation locks
    for team, p in sorted(rel_probs.items(), key=lambda x: x[1], reverse=True)[:3]:
        if p >= 0.65:
            bets.append({
                "market":    "Relegation",
                "selection": sname(team),
                "team":      team,
                "prob":      p,
                "why":       f"{p*100:.0f}% probability of finishing in the bottom 3.",
            })

    # Upcoming match bets — take highest single-outcome probability per match
    for fix in upcoming_fixtures[:8]:
        outcomes = [
            ("Home Win",  fix["home_win"],  fix["home_team"]),
            ("Draw",      fix["draw"],      None),
            ("Away Win",  fix["away_win"],  fix["away_team"]),
        ]
        best_lbl, best_p, best_t = max(outcomes, key=lambda x: x[1])
        label_team = sname(best_t) if best_t else "Draw"
        h  = sname(fix["home_team"])
        a  = sname(fix["away_team"])
        dt = fix["date"][:10]
        if best_p >= 0.48:
            bets.append({
                "market":    f"Match Result ({dt})",
                "selection": f"{h} vs {a} — {best_lbl}",
                "team":      best_t or fix["home_team"],
                "prob":      best_p,
                "why":       f"xG {fix['xg_home']:.1f}–{fix['xg_away']:.1f}. "
                             f"Over 2.5: {fix['over_25']*100:.0f}% | BTTS: {fix['btts']*100:.0f}%",
            })

    # Sort by probability (best edge first)
    bets.sort(key=lambda b: b["prob"], reverse=True)

    cards = []
    for rank, bet in enumerate(bets[:12], 1):
        p    = bet["prob"]
        c    = tcolor(bet["team"])
        lbl, lc = confidence(p)
        logo = _logo_img(bet["team"], 32)
        medal = ["", "#FFD700", "#C0C0C0", "#CD7F32"]
        medal_c = medal[rank] if rank <= 3 else "#555"
        cards.append(f"""
<div style="display:flex;align-items:flex-start;gap:14px;padding:14px 18px;
            border-radius:10px;background:rgba(255,255,255,0.03);
            border:1px solid rgba(255,255,255,0.07);
            border-left:4px solid {c};margin-bottom:10px;">
  <div style="font-size:1.4rem;font-weight:900;color:{medal_c};min-width:28px;
              text-align:center;padding-top:2px;">#{rank}</div>
  {logo}
  <div style="flex:1;">
    <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:4px;">
      <span style="font-size:0.7rem;text-transform:uppercase;letter-spacing:1px;
                   color:#888;background:rgba(255,255,255,0.06);padding:2px 8px;
                   border-radius:4px;">{bet['market']}</span>
      <span style="font-weight:800;font-size:0.95rem;color:white;">{bet['selection']}</span>
      <span style="background:{lc};color:#000;font-size:0.68rem;font-weight:800;
                   padding:2px 7px;border-radius:4px;">{lbl}</span>
    </div>
    <div style="font-size:0.82rem;color:#aaa;margin-bottom:6px;">{bet['why']}</div>
    <div style="display:flex;gap:20px;flex-wrap:wrap;">
      <div><span style="color:#666;font-size:0.75rem;">MODEL PROB</span><br>
           <span style="font-size:1.2rem;font-weight:900;color:{c};">{p*100:.1f}%</span></div>
      <div><span style="color:#666;font-size:0.75rem;">FAIR ODDS</span><br>
           <span style="font-size:1.2rem;font-weight:700;color:#ddd;">{odds(p)}</span></div>
      <div style="max-width:200px;"><span style="color:#666;font-size:0.75rem;">BETTING TIP</span><br>
           <span style="font-size:0.8rem;color:#c084fc;">
             Seek odds &gt; {odds(p)} for positive EV
           </span></div>
    </div>
  </div>
</div>""")

    return "\n".join(cards)


# ── Upcoming fixtures table ───────────────────────────────────────────────────

def _fixture_cards(upcoming_fixtures: list) -> str:
    cards = []
    for fix in upcoming_fixtures:
        h = sname(fix["home_team"]); a = sname(fix["away_team"])
        hc = tcolor(fix["home_team"]); ac = tcolor(fix["away_team"])
        hw = fix["home_win"]; dr = fix["draw"]; aw = fix["away_win"]
        xgh = fix["xg_home"]; xga = fix["xg_away"]
        o25 = fix["over_25"]; btts = fix["btts"]
        dt  = fix["date"][:10]

        best_p = max(hw, dr, aw)
        if best_p == hw:
            rec = f"<span style='color:{hc};font-weight:700;'>{h} to win</span>"
        elif best_p == dr:
            rec = "<span style='color:#f59e0b;font-weight:700;'>Draw</span>"
        else:
            rec = f"<span style='color:{ac};font-weight:700;'>{a} to win</span>"

        clbl, clc = confidence(best_p)

        def pct_bar(p, c, w=60):
            return (f'<div style="font-weight:700;font-size:0.9rem;color:{c};">{p*100:.0f}%</div>'
                    f'<div style="width:{w}px;height:5px;background:rgba(255,255,255,0.08);'
                    f'border-radius:3px;margin-top:2px;">'
                    f'<div style="width:{p*w:.0f}px;height:5px;background:{c};border-radius:3px;"></div></div>')

        cards.append(f"""
<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
            border-radius:10px;padding:16px 20px;margin-bottom:10px;">

  <!-- Teams row -->
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;flex-wrap:wrap;gap:8px;">
    <div style="display:flex;align-items:center;gap:8px;">
      {_logo_img(fix['home_team'],34)}
      <span style="font-size:1rem;font-weight:700;color:{hc};">{h}</span>
    </div>
    <div style="text-align:center;">
      <div style="font-size:0.72rem;color:#666;letter-spacing:1px;">VS &nbsp; {dt}</div>
      <div style="font-size:0.78rem;color:#888;margin-top:2px;">
        xG: <b style="color:{hc};">{xgh:.2f}</b> – <b style="color:{ac};">{xga:.2f}</b>
      </div>
    </div>
    <div style="display:flex;align-items:center;gap:8px;">
      <span style="font-size:1rem;font-weight:700;color:{ac};">{a}</span>
      {_logo_img(fix['away_team'],34)}
    </div>
  </div>

  <!-- Probability row -->
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1px 1fr 1fr;
              gap:10px;align-items:start;">
    <div style="text-align:center;">
      <div style="font-size:0.68rem;color:#666;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:4px;">Home Win</div>
      {pct_bar(hw, hc)}
    </div>
    <div style="text-align:center;">
      <div style="font-size:0.68rem;color:#666;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:4px;">Draw</div>
      {pct_bar(dr, '#f59e0b')}
    </div>
    <div style="text-align:center;">
      <div style="font-size:0.68rem;color:#666;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:4px;">Away Win</div>
      {pct_bar(aw, ac)}
    </div>
    <div style="background:rgba(255,255,255,0.08);width:1px;"></div>
    <div style="text-align:center;">
      <div style="font-size:0.68rem;color:#666;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:4px;">Over 2.5</div>
      {pct_bar(o25, '#818cf8')}
    </div>
    <div style="text-align:center;">
      <div style="font-size:0.68rem;color:#666;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:4px;">BTTS</div>
      {pct_bar(btts, '#34d399')}
    </div>
  </div>

  <!-- Recommendation -->
  <div style="margin-top:10px;padding-top:10px;border-top:1px solid rgba(255,255,255,0.05);
              display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
    <span style="color:#666;font-size:0.78rem;">Best bet:</span>
    {rec}
    <span style="color:#aaa;font-size:0.8rem;">({best_p*100:.0f}% · fair odds {odds(best_p)})</span>
    <span style="background:{clc};color:#000;font-size:0.68rem;font-weight:800;
                 padding:1px 7px;border-radius:4px;">{clbl}</span>
  </div>
</div>""")

    return "\n".join(cards)


# ── Elo chart ─────────────────────────────────────────────────────────────────

def _elo_chart_html(elo_history: pd.DataFrame, elo_ratings: dict) -> str:
    import plotly.io as pio
    top8 = [t for t, _ in sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)[:8]]
    fig  = go.Figure()
    for team in top8:
        df = elo_history[elo_history["team"] == team].sort_values("date")
        if df.empty: continue
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["rating"], mode="lines",
            name=sname(team), line=dict(color=tcolor(team), width=2.5),
            hovertemplate=f"<b>{sname(team)}</b><br>%{{x|%d %b %Y}}<br>Elo: %{{y:.0f}}<extra></extra>",
        ))
    fig.update_layout(
        paper_bgcolor=PANEL_BG, plot_bgcolor=PANEL_BG,
        font=dict(color="white", family="Segoe UI, Arial"),
        height=380, margin=dict(l=60, r=20, t=40, b=50),
        legend=dict(font=dict(color="white", size=10),
                    bgcolor="rgba(20,20,40,0.8)",
                    bordercolor="rgba(255,255,255,0.12)", borderwidth=1),
        xaxis=dict(gridcolor="rgba(255,255,255,0.07)", tickfont=dict(color="#aaa")),
        yaxis=dict(title="Elo Rating", title_font=dict(color="#aaa", size=11),
                   gridcolor="rgba(255,255,255,0.07)", tickfont=dict(color="#aaa")),
    )
    return pio.to_html(fig, full_html=False, include_plotlyjs="cdn",
                       config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: #0D0D1E; color: #f0f0f0;
  font-family: 'Segoe UI', Arial, sans-serif; min-height: 100vh;
}
.site-header {
  background: linear-gradient(135deg, #180030 0%, #3d006a 50%, #180030 100%);
  border-bottom: 3px solid #7c3aed;
  padding: 20px 34px; display: flex; align-items: center; gap: 16px;
}
.pl-badge {
  width: 48px; height: 48px; background: #7c3aed; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 19px; font-weight: 900; color: white; flex-shrink: 0;
  box-shadow: 0 0 22px rgba(124,58,237,0.7); letter-spacing: -1px;
}
.header-text h1 {
  font-size: 1.7rem; font-weight: 900;
  background: linear-gradient(90deg, #fff, #c084fc);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.header-text p { color: #a78bfa; font-size: 0.85rem; margin-top: 3px; }
.main { padding: 22px 30px; }
.winner-card { border-radius: 14px; padding: 24px 28px;
  display: flex; align-items: center; gap: 24px; margin-bottom: 22px; }
.winner-info h2 { font-size: 0.82rem; text-transform: uppercase;
  letter-spacing: 1.8px; color: #aaa; margin-bottom: 5px; }
.winner-name { font-size: 2.5rem; font-weight: 900; line-height: 1.1; }
.winner-pct { font-size: 1.05rem; color: #ccc; margin-top: 5px; }
.runners { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px; }
.runner { background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1);
  border-radius: 8px; padding: 6px 11px;
  display: flex; align-items: center; gap: 7px; font-size: 0.84rem; }
.runner-pct { color: #888; font-size: 0.76rem; }
.card { background: #14142B; border: 1px solid rgba(255,255,255,0.07);
  border-radius: 12px; overflow: hidden; margin-bottom: 22px; }
.card-header { padding: 12px 18px; font-size: 0.71rem; font-weight: 700;
  letter-spacing: 1.4px; text-transform: uppercase; color: #6666aa;
  border-bottom: 1px solid rgba(255,255,255,0.06); }
.card-body { padding: 14px; }
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 22px; }
.three-col { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
@media (max-width: 1000px) { .two-col { grid-template-columns: 1fr; } }
@media (max-width: 800px)  { .three-col { grid-template-columns: 1fr; } }
.standings-tbl { width: 100%; border-collapse: collapse; font-size: 0.84rem; }
.standings-tbl th { padding: 7px 9px; text-align: left; font-size: 0.68rem;
  letter-spacing: 1px; text-transform: uppercase; color: #555577;
  border-bottom: 1px solid rgba(255,255,255,0.07); }
.standings-tbl td { padding: 5px 9px; border-bottom: 1px solid rgba(255,255,255,0.03);
  vertical-align: middle; }
.standings-tbl tr:hover td { background: rgba(255,255,255,0.03); }
.market-tbl { width: 100%; border-collapse: collapse; font-size: 0.84rem; }
.market-tbl th { padding: 7px 10px; text-align: left; font-size: 0.68rem;
  letter-spacing: 1px; text-transform: uppercase; color: #555577;
  border-bottom: 1px solid rgba(255,255,255,0.07); }
.market-tbl td { padding: 7px 10px; border-bottom: 1px solid rgba(255,255,255,0.04);
  vertical-align: middle; }
.legend { display: flex; gap: 14px; padding: 9px 16px; font-size: 0.7rem;
  color: #666; border-top: 1px solid rgba(255,255,255,0.05); }
.pill { width: 9px; height: 9px; border-radius: 2px; display: inline-block;
  margin-right: 4px; vertical-align: middle; }
.callout { background: rgba(124,58,237,0.09); border: 1px solid rgba(124,58,237,0.28);
  border-radius: 10px; padding: 13px 18px; font-size: 0.8rem; color: #c4b5fd;
  line-height: 1.65; margin-bottom: 22px; }
.callout strong { color: #ddd; }
"""


# ---------------------------------------------------------------------------
# Master HTML builder
# ---------------------------------------------------------------------------

def build_html(
    win_probs, top4_probs, rel_probs,
    cur_points, cur_gd, gf, elo_ratings,
    form_scores, form_strings,
    teams, target_season, remaining_count, is_complete,
    upcoming_fixtures: list,
    elo_chart_html: str,
) -> str:
    winner     = max(win_probs, key=win_probs.get)
    winner_p   = win_probs[winner]
    w_color    = tcolor(winner)
    w_logo_url = tlogo(winner)
    w_short    = sname(winner)
    season_lbl = target_season.replace("-", "/")
    match_lbl  = "Season complete" if is_complete else f"{remaining_count} fixtures remaining"

    logo_big = (f'<img src="{w_logo_url}" width="100" height="100" '
                f'style="filter:drop-shadow(0 0 20px {w_color});object-fit:contain;" '
                f'onerror="this.style.display=\'none\'">'
                if w_logo_url else "")

    by_prob = sorted(win_probs.items(), key=lambda x: x[1], reverse=True)
    runners_html = ""
    for t, p in by_prob[1:4]:
        c  = tcolor(t); lg = tlogo(t)
        lgi = (f'<img src="{lg}" width="26" height="26" style="object-fit:contain;" '
               f'onerror="this.style.display:none">' if lg else "")
        runners_html += (f'<div class="runner">{lgi}'
                         f'<span style="color:{c};font-weight:700;">{sname(t)}</span>'
                         f'<span class="runner-pct">{p*100:.1f}%</span></div>')

    prob_bars   = _prob_bars_html(win_probs, cur_points, cur_gd, elo_ratings, form_strings)
    standings   = _standings_table(teams, cur_points, cur_gd, gf, elo_ratings, win_probs, form_strings)
    best_bets   = _best_bets_cards(win_probs, top4_probs, rel_probs, upcoming_fixtures)
    fixture_sec = _fixture_cards(upcoming_fixtures)
    top4_rows   = _season_market_rows("Top 4", top4_probs, top_n=6)
    rel_rows    = _season_market_rows("Relegation", rel_probs, top_n=5)
    title_rows  = _season_market_rows("Title", win_probs, top_n=4)

    wrgb = _hex_rgb(w_color)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>PL {season_lbl} — Prediction & Betting Guide</title>
<style>{CSS}</style>
</head>
<body>

<header class="site-header">
  <div class="pl-badge">PL</div>
  <div class="header-text">
    <h1>Premier League {season_lbl} — Prediction &amp; Betting Guide</h1>
    <p>Dixon-Coles Poisson model &nbsp;·&nbsp; {match_lbl} &nbsp;·&nbsp; 10,000 Monte Carlo simulations</p>
  </div>
</header>

<div class="main">

  <div class="callout">
    <strong>How this works:</strong> The model fits separate attack and defence strength parameters
    for every club from all historical results (time-weighted so recent games count more), then
    simulates each remaining fixture by sampling goals from a Poisson distribution.
    Win probabilities, Over/Under, BTTS and positional market odds are all derived from the same model.
    <strong>Fair odds</strong> shown below = 1 / model probability. Any bookmaker price <em>higher</em>
    than the fair odds represents a positive expected-value (+EV) bet.
  </div>

  <!-- Winner spotlight -->
  <div class="winner-card" style="
    background:linear-gradient(135deg,{CARD_BG} 0%,rgba({wrgb},0.17) 100%);
    border:1px solid rgba({wrgb},0.4);
    box-shadow:0 0 50px rgba({wrgb},0.14);">
    <div style="flex-shrink:0;">{logo_big}</div>
    <div class="winner-info">
      <h2>Predicted Champion</h2>
      <div class="winner-name" style="color:{w_color};
        text-shadow:0 0 40px rgba({wrgb},0.5);">{w_short}</div>
      <div class="winner-pct">
        <span style="font-size:1.55rem;font-weight:900;color:{w_color};">{winner_p*100:.1f}%</span>
        title probability &nbsp;·&nbsp; fair odds {odds(winner_p)}
      </div>
      <div class="runners">
        <span style="color:#666;font-size:0.8rem;align-self:center;">Others in contention:</span>
        {runners_html}
      </div>
    </div>
  </div>

  <!-- Prob bars + standings -->
  <div class="two-col">
    <div class="card">
      <div class="card-header">Title Win Probability</div>
      <div class="card-body" style="padding:10px 6px;">{prob_bars}</div>
    </div>
    <div class="card">
      <div class="card-header">Current Standings</div>
      <div style="overflow-y:auto;max-height:640px;">
        <table class="standings-tbl">
          <thead><tr><th>#</th><th>Club</th><th>Pts</th><th>GD</th>
            <th>Elo</th><th>Form</th><th>Title %</th></tr></thead>
          <tbody>{standings}</tbody>
        </table>
      </div>
      <div class="legend">
        <span><span class="pill" style="background:rgba(255,215,0,0.5)"></span>Leader</span>
        <span><span class="pill" style="background:rgba(59,130,246,0.5)"></span>UCL (Top 4)</span>
        <span><span class="pill" style="background:rgba(239,68,68,0.4)"></span>Relegation</span>
      </div>
    </div>
  </div>

  <!-- Season markets -->
  <div class="card">
    <div class="card-header">Season Markets — Model Probabilities &amp; Fair Odds</div>
    <div class="card-body">
      <div class="three-col">
        <div>
          <div style="font-size:0.75rem;text-transform:uppercase;letter-spacing:1px;
                      color:#888;margin-bottom:10px;">Title Winner</div>
          <table class="market-tbl">
            <thead><tr><th>Club</th><th>Prob</th><th>Fair Odds</th><th></th></tr></thead>
            <tbody>{title_rows}</tbody>
          </table>
        </div>
        <div>
          <div style="font-size:0.75rem;text-transform:uppercase;letter-spacing:1px;
                      color:#888;margin-bottom:10px;">Top 4 Finish (UCL)</div>
          <table class="market-tbl">
            <thead><tr><th>Club</th><th>Prob</th><th>Fair Odds</th><th></th></tr></thead>
            <tbody>{top4_rows}</tbody>
          </table>
        </div>
        <div>
          <div style="font-size:0.75rem;text-transform:uppercase;letter-spacing:1px;
                      color:#888;margin-bottom:10px;">Relegation (Bottom 3)</div>
          <table class="market-tbl">
            <thead><tr><th>Club</th><th>Prob</th><th>Fair Odds</th><th></th></tr></thead>
            <tbody>{rel_rows}</tbody>
          </table>
        </div>
      </div>
    </div>
  </div>

  <!-- Best bets -->
  <div class="card">
    <div class="card-header">Best Bets — Ranked by Model Confidence</div>
    <div class="card-body">{best_bets}</div>
  </div>

  <!-- Upcoming fixtures -->
  <div class="card">
    <div class="card-header">Upcoming Fixture Predictions</div>
    <div class="card-body">{fixture_sec}</div>
  </div>

  <!-- Elo chart -->
  <div class="card">
    <div class="card-header">Elo Rating Trajectory — Top 8 Teams (All Seasons)</div>
    <div style="padding:4px;">{elo_chart_html}</div>
  </div>

</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs("data", exist_ok=True)
    data_path = "data/matches.json"
    if not os.path.exists(data_path):
        print("Fetching data first...")
        import subprocess
        subprocess.run(["py", os.path.join(os.path.dirname(__file__), "data_fetch.py")], check=True)

    print("Loading match data...")
    completed, upcoming = load_season_data(data_path)
    if completed.empty:
        print("ERROR: No completed matches found."); return

    seasons_with_data = completed.groupby("season").size().sort_index().index.tolist()
    upcoming_seasons  = upcoming["season"].unique().tolist() if not upcoming.empty else []
    all_seasons       = sorted(set(seasons_with_data + upcoming_seasons))
    target_season     = all_seasons[-1]
    if target_season not in seasons_with_data:
        target_season = seasons_with_data[-1]

    s_done = completed[completed["season"] == target_season]
    s_todo = upcoming[upcoming["season"] == target_season] if not upcoming.empty else pd.DataFrame()
    remaining_count = len(s_todo)
    is_complete     = (remaining_count == 0)

    print(f"Target season : {target_season}")
    print(f"Completed     : {len(s_done)} matches")
    print(f"Remaining     : {remaining_count} fixtures")

    print("Running Monte Carlo (50,000 iterations, v3 model)...")
    (win_probs, top4_probs, rel_probs,
     cur_points, cur_gd, elo_ratings, teams,
     atk_home, atk_away, def_home, def_away, home_adv, rho,
     form_atk, form_def) = run_monte_carlo(
        completed, upcoming, target_season
    )

    _, _, gf = compute_standings(completed, target_season)
    _, _, form_strings = compute_goals_form(completed, target_season)

    # Upcoming fixtures with full probability breakdown
    today_str = str(date.today())
    print("Computing upcoming fixture probabilities...")
    upcoming_fixtures = predict_upcoming_fixtures(
        upcoming, completed, target_season,
        atk_home, atk_away, def_home, def_away,
        home_adv, rho, form_atk, form_def,
        n=10, from_date=today_str,
    )
    if not upcoming_fixtures:
        upcoming_fixtures = predict_upcoming_fixtures(
            upcoming, completed, target_season,
            atk_home, atk_away, def_home, def_away,
            home_adv, rho, form_atk, form_def,
            n=10, from_date=None,
        )

    # Console output
    print("\nTitle probabilities:")
    for t, p in sorted(win_probs.items(), key=lambda x: x[1], reverse=True):
        print(f"  {sname(t):<22} {p*100:6.2f}%  (fair odds {odds(p)})")

    print("\nTop-4 probabilities (top 6):")
    for t, p in sorted(top4_probs.items(), key=lambda x: x[1], reverse=True)[:6]:
        print(f"  {sname(t):<22} {p*100:6.1f}%")

    print("\nRelegation probabilities (top 5):")
    for t, p in sorted(rel_probs.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {sname(t):<22} {p*100:6.1f}%")

    print("\nBuilding dashboard...")
    elo_history    = compute_elo_history(completed)
    elo_chart_html = _elo_chart_html(elo_history, elo_ratings)

    html = build_html(
        win_probs, top4_probs, rel_probs,
        cur_points, cur_gd, gf, elo_ratings,
        form_atk, form_strings,
        teams, target_season, remaining_count, is_complete,
        upcoming_fixtures, elo_chart_html,
    )

    out = "data/prediction_dashboard.html"
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)

    abs_path = os.path.abspath(out).replace(os.sep, "/")
    print(f"\nDashboard saved: {abs_path}")
    print("Opening in browser...")
    webbrowser.open(f"file:///{abs_path}")


if __name__ == "__main__":
    main()
