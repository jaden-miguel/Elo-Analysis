"""
Premier League Analysis Hub — Full Dashboard
=============================================
Generates data/prediction_dashboard.html with:
  - Tabbed navigation (Title Race · Season Race · Scout · Top Scorers · Betting Edge)
  - Dixon-Coles win-probability bars and standings
  - Animated bar-chart race replaying the season matchday-by-matchday
  - Player scouting section with breakout scores
  - Goal scorer predictions from the Dixon-Coles model
  - Polymarket betting edge analysis with EV, Kelly, Z-score

Run:  python src/predict_dashboard.py
"""

import json
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
    compute_goals_form,
    compute_matchday_standings,
    predict_remaining_goals,
    fit_dixoncoles,
    _simple_elo,
)
from player_data import PL_PLAYERS, compute_breakout_scores
from polymarket_data import fetch_all_soccer_markets, parse_markets, analyze_all_edges
from league_models import LEAGUES

# ---------------------------------------------------------------------------
# Team identity
# ---------------------------------------------------------------------------

_PL100 = "https://resources.premierleague.com/premierleague/badges/100"

TEAM_COLORS: dict[str, str] = {
    "Arsenal": "#EF0107", "Aston Villa": "#7B003C", "Bournemouth": "#DA291C",
    "Brentford": "#E30613", "Brighton": "#0057B8", "Chelsea": "#034694",
    "Crystal Palace": "#1B458F", "Everton": "#003399", "Fulham": "#CC0000",
    "Ipswich": "#0044A9", "Leicester": "#003090", "Liverpool": "#C8102E",
    "Man City": "#6CABDD", "Man Utd": "#DA291C", "Newcastle": "#241F20",
    "Nottm Forest": "#DD0000", "Southampton": "#D71920", "Tottenham": "#132257",
    "West Ham": "#7A263A", "Wolves": "#FDB913", "Burnley": "#6C1D45",
    "Leeds": "#FFCD00", "Sheffield Utd": "#EE2737", "Sunderland": "#EB172B",
    "Luton": "#F78F1E", "Norwich": "#00A650", "Middlesbrough": "#E01A17",
}

TEAM_LOGOS: dict[str, str] = {
    "Arsenal": f"{_PL100}/t3.png", "Aston Villa": f"{_PL100}/t7.png",
    "Bournemouth": f"{_PL100}/t91.png", "Brentford": f"{_PL100}/t94.png",
    "Brighton": f"{_PL100}/t36.png", "Chelsea": f"{_PL100}/t8.png",
    "Crystal Palace": f"{_PL100}/t31.png", "Everton": f"{_PL100}/t11.png",
    "Fulham": f"{_PL100}/t54.png", "Ipswich": f"{_PL100}/t40.png",
    "Leicester": f"{_PL100}/t13.png", "Liverpool": f"{_PL100}/t14.png",
    "Man City": f"{_PL100}/t43.png", "Man Utd": f"{_PL100}/t1.png",
    "Newcastle": f"{_PL100}/t4.png", "Nottm Forest": f"{_PL100}/t17.png",
    "Southampton": f"{_PL100}/t20.png", "Tottenham": f"{_PL100}/t6.png",
    "West Ham": f"{_PL100}/t21.png", "Wolves": f"{_PL100}/t39.png",
    "Burnley": f"{_PL100}/t90.png", "Leeds": f"{_PL100}/t2.png",
    "Sheffield Utd": f"{_PL100}/t49.png", "Sunderland": f"{_PL100}/t56.png",
    "Luton": f"{_PL100}/t102.png",
}

_ALIASES: dict[str, str] = {
    "Arsenal FC": "Arsenal", "AFC Bournemouth": "Bournemouth",
    "Aston Villa FC": "Aston Villa", "Brentford FC": "Brentford",
    "Brighton & Hove Albion FC": "Brighton", "Chelsea FC": "Chelsea",
    "Crystal Palace FC": "Crystal Palace", "Everton FC": "Everton",
    "Fulham FC": "Fulham", "Ipswich Town FC": "Ipswich",
    "Leicester City FC": "Leicester", "Liverpool FC": "Liverpool",
    "Manchester City FC": "Man City", "Manchester United FC": "Man Utd",
    "Newcastle United FC": "Newcastle", "Nottingham Forest FC": "Nottm Forest",
    "Southampton FC": "Southampton", "Tottenham Hotspur FC": "Tottenham",
    "West Ham United FC": "West Ham", "Wolverhampton Wanderers FC": "Wolves",
    "Burnley FC": "Burnley", "Leeds United FC": "Leeds",
    "Sheffield United FC": "Sheffield Utd", "Sunderland AFC": "Sunderland",
    "Luton Town FC": "Luton",
    # without FC
    "Arsenal": "Arsenal", "AFC Bournemouth": "Bournemouth",
    "Aston Villa": "Aston Villa", "Brentford": "Brentford",
    "Brighton & Hove Albion": "Brighton", "Chelsea": "Chelsea",
    "Crystal Palace": "Crystal Palace", "Everton": "Everton",
    "Fulham": "Fulham", "Ipswich Town": "Ipswich",
    "Leicester City": "Leicester", "Liverpool": "Liverpool",
    "Manchester City": "Man City", "Manchester United": "Man Utd",
    "Newcastle United": "Newcastle", "Nottingham Forest": "Nottm Forest",
    "Southampton": "Southampton", "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham", "Wolverhampton Wanderers": "Wolves",
    "Burnley": "Burnley", "Leeds United": "Leeds",
    "Sheffield United": "Sheffield Utd", "Sunderland": "Sunderland",
    "Luton Town": "Luton",
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

def _hex_to_rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = h[0]*2 + h[1]*2 + h[2]*2
    try:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"{r},{g},{b}"
    except Exception:
        return "74,144,226"


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = r"""
*{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{background:#0D0D1E;color:#f0f0f0;font-family:'Inter','Segoe UI',Arial,sans-serif;min-height:100vh}

/* Custom scrollbar */
::-webkit-scrollbar{width:6px;height:6px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:rgba(124,58,237,.35);border-radius:3px}
::-webkit-scrollbar-thumb:hover{background:rgba(124,58,237,.55)}

/* ── Header ─────────────────────────────────────────────── */
.site-header{
  background:linear-gradient(135deg,#180030 0%,#3d006a 50%,#180030 100%);
  border-bottom:3px solid #7c3aed;padding:20px 36px;
  display:flex;align-items:center;gap:18px;flex-wrap:wrap;
  position:relative;overflow:hidden;
}
.site-header::before{
  content:'';position:absolute;inset:0;
  background:radial-gradient(ellipse at 70% 50%,rgba(124,58,237,.15),transparent 60%);
  pointer-events:none;
}
.pl-badge{
  width:50px;height:50px;background:#7c3aed;border-radius:50%;
  display:flex;align-items:center;justify-content:center;
  font-size:20px;font-weight:900;color:white;flex-shrink:0;
  box-shadow:0 0 22px rgba(124,58,237,.7);letter-spacing:-1px;
  animation:badgePulse 3s ease-in-out infinite;
  position:relative;z-index:1;
}
@keyframes badgePulse{
  0%,100%{box-shadow:0 0 22px rgba(124,58,237,.7)}
  50%{box-shadow:0 0 32px rgba(124,58,237,.9),0 0 60px rgba(124,58,237,.3)}
}
.header-text{position:relative;z-index:1}
.header-text h1{
  font-size:1.6rem;font-weight:900;
  background:linear-gradient(90deg,#fff,#c084fc);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
}
.header-text p{color:#a78bfa;font-size:.82rem;margin-top:4px}

/* ── Tab navigation ─────────────────────────────────────── */
.tab-nav{
  display:flex;gap:0;background:#0f0f24;
  border-bottom:1px solid rgba(255,255,255,.06);
  padding:0 32px;overflow-x:auto;
  position:sticky;top:0;z-index:100;
  backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);
}
.tab-btn{
  background:none;border:none;color:#666;font-size:.82rem;
  font-weight:600;padding:14px 22px;cursor:pointer;
  position:relative;white-space:nowrap;
  letter-spacing:.4px;
  transition:color .25s,background .25s;
  border-radius:6px 6px 0 0;
}
.tab-btn:hover{color:#c084fc;background:rgba(124,58,237,.06)}
.tab-btn:active{transform:scale(.97)}
.tab-btn.active{color:#c084fc}
.tab-btn.active::after{
  content:'';position:absolute;bottom:0;left:12px;right:12px;
  height:2px;background:linear-gradient(90deg,#7c3aed,#c084fc);
  border-radius:1px;
  animation:tabSlide .3s ease;
}
@keyframes tabSlide{from{transform:scaleX(0)}to{transform:scaleX(1)}}

/* ── Sections ───────────────────────────────────────────── */
.tab-section{display:none;animation:sectionIn .4s cubic-bezier(.4,0,.2,1)}
.tab-section.active{display:block}
@keyframes sectionIn{
  from{opacity:0;transform:translateY(12px)}
  to{opacity:1;transform:none}
}
.main{padding:24px 32px;max-width:1440px;margin:0 auto}

/* ── Cards ──────────────────────────────────────────────── */
.card{
  background:rgba(20,20,43,.85);
  backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);
  border:1px solid rgba(255,255,255,.07);
  border-radius:14px;overflow:hidden;margin-bottom:24px;
  transition:border-color .3s,box-shadow .3s;
}
.card:hover{
  border-color:rgba(124,58,237,.18);
  box-shadow:0 4px 24px rgba(0,0,0,.3);
}
.card-header{
  padding:13px 18px;font-size:.73rem;font-weight:700;
  letter-spacing:1.4px;text-transform:uppercase;color:#6666aa;
  border-bottom:1px solid rgba(255,255,255,.06);
  display:flex;align-items:center;justify-content:space-between;
}
.card-body{padding:16px}

/* ── Two-column ─────────────────────────────────────────── */
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:22px;margin-bottom:24px}
@media(max-width:960px){.two-col{grid-template-columns:1fr}}

/* ── Winner spotlight ───────────────────────────────────── */
.winner-card{
  border-radius:14px;padding:26px 30px;display:flex;align-items:center;gap:26px;margin-bottom:24px;
  position:relative;overflow:hidden;
}
.winner-card::before{
  content:'';position:absolute;top:-40%;right:-20%;width:300px;height:300px;
  border-radius:50%;background:radial-gradient(circle,rgba(255,255,255,.04),transparent 70%);
  pointer-events:none;
}
.winner-info h2{font-size:.85rem;text-transform:uppercase;letter-spacing:1.8px;color:#aaa;margin-bottom:6px}
.winner-name{font-size:2.6rem;font-weight:900;line-height:1.1}
.winner-pct{font-size:1.1rem;color:#ccc;margin-top:6px}
.runners{display:flex;gap:12px;flex-wrap:wrap;margin-top:12px}
.runner{
  background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.1);
  border-radius:8px;padding:7px 12px;display:flex;align-items:center;
  gap:8px;font-size:.87rem;
  transition:transform .2s,background .2s;
}
.runner:hover{transform:translateY(-2px);background:rgba(255,255,255,.09)}
.runner-pct{color:#888;font-size:.78rem}

/* ── Standings table ────────────────────────────────────── */
.standings-tbl{width:100%;border-collapse:collapse;font-size:.85rem}
.standings-tbl th{
  padding:8px 10px;text-align:left;font-size:.7rem;
  letter-spacing:1px;text-transform:uppercase;color:#555577;
  border-bottom:1px solid rgba(255,255,255,.07);
  position:sticky;top:0;background:#14142B;z-index:1;
}
.standings-tbl td{
  padding:6px 10px;border-bottom:1px solid rgba(255,255,255,.03);
  vertical-align:middle;transition:background .2s;
}
.standings-tbl tr:hover td{background:rgba(255,255,255,.04)}

/* ── Legend ──────────────────────────────────────────────── */
.legend{
  display:flex;gap:16px;padding:10px 18px;
  font-size:.72rem;color:#666;border-top:1px solid rgba(255,255,255,.05);
}
.pill{width:10px;height:10px;border-radius:2px;display:inline-block;margin-right:5px;vertical-align:middle}

/* ── Model callout ──────────────────────────────────────── */
.model-callout{
  background:rgba(124,58,237,.08);border:1px solid rgba(124,58,237,.25);
  border-radius:10px;padding:14px 20px;font-size:.82rem;color:#c4b5fd;
  line-height:1.6;margin-bottom:24px;
}
.model-callout strong{color:#ddd}

/* ── Race animation ─────────────────────────────────────── */
.race-header{
  display:flex;align-items:center;justify-content:space-between;
  flex-wrap:wrap;gap:12px;margin-bottom:16px;
}
.race-matchday{
  font-size:3.2rem;font-weight:900;
  background:linear-gradient(135deg,#7c3aed,#c084fc);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  line-height:1;
  transition:transform .3s;
}
.race-matchday.bump{animation:mwBump .35s ease}
@keyframes mwBump{
  0%{transform:scale(1)}
  40%{transform:scale(1.12)}
  100%{transform:scale(1)}
}
.race-date{font-size:.85rem;color:#888;margin-top:4px}
.race-progress-track{
  width:100%;height:5px;background:rgba(255,255,255,.06);
  border-radius:3px;margin-bottom:20px;overflow:hidden;
}
.race-progress-fill{
  height:100%;border-radius:3px;
  background:linear-gradient(90deg,#7c3aed,#c084fc);
  transition:width .45s ease;
  box-shadow:0 0 8px rgba(124,58,237,.5);
}
.race-container{position:relative;width:100%;overflow:hidden}
.race-bar{
  position:absolute;left:0;width:100%;height:36px;
  display:flex;align-items:center;gap:8px;padding:0 8px;
  transition:transform .55s cubic-bezier(.4,0,.2,1);
  will-change:transform;
}
.race-bar img{width:26px;height:26px;object-fit:contain;flex-shrink:0;border-radius:3px}
.race-bar .rname{
  width:110px;flex-shrink:0;font-size:.78rem;font-weight:600;
  color:#ccc;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
  transition:color .3s,font-weight .3s;
}
.race-bar .rfill{
  flex:1;height:22px;border-radius:5px;
  transition:width .55s cubic-bezier(.4,0,.2,1);min-width:2px;
  position:relative;
}
.race-bar .rpts{
  min-width:40px;text-align:right;font-size:.85rem;font-weight:800;
  color:#fff;
  font-variant-numeric:tabular-nums;
  transition:color .3s,transform .35s;
}
.race-bar .rpos{
  width:22px;font-size:.65rem;font-weight:700;text-align:center;
  color:#555;flex-shrink:0;transition:color .3s;
}
.race-bar .rpos.up{color:#22c55e}
.race-bar .rpos.down{color:#ef4444}
.race-bar .rpos.leader{color:#f59e0b}

/* Race controls — large, tactile, responsive */
.race-controls{
  display:flex;align-items:center;justify-content:center;
  gap:10px;padding:22px 0 10px;flex-wrap:wrap;
}
.race-btn{
  background:rgba(124,58,237,.15);
  border:1px solid rgba(124,58,237,.35);
  color:#c084fc;border-radius:10px;
  padding:12px 24px;cursor:pointer;
  font-size:.85rem;font-weight:700;
  transition:all .15s ease;
  position:relative;overflow:hidden;
  user-select:none;-webkit-user-select:none;
  outline:none;
}
.race-btn:hover{
  background:rgba(124,58,237,.28);
  border-color:rgba(124,58,237,.55);
  transform:translateY(-1px);
  box-shadow:0 4px 16px rgba(124,58,237,.2);
}
.race-btn:active{
  transform:translateY(0) scale(.96);
  box-shadow:none;
  background:rgba(124,58,237,.35);
  transition-duration:.05s;
}
.race-btn:focus-visible{
  outline:2px solid #c084fc;outline-offset:2px;
}
.race-btn.play-main{
  min-width:140px;
  background:linear-gradient(135deg,#7c3aed,#6d28d9);
  border-color:#7c3aed;
  color:white;
  font-size:.9rem;
  padding:14px 30px;
  box-shadow:0 4px 20px rgba(124,58,237,.3);
}
.race-btn.play-main:hover{
  box-shadow:0 6px 28px rgba(124,58,237,.45);
  transform:translateY(-2px);
}
.race-btn.play-main:active{
  transform:translateY(0) scale(.96);
  box-shadow:0 2px 10px rgba(124,58,237,.2);
}
.race-btn.play-main.is-playing{
  background:linear-gradient(135deg,#ef4444,#dc2626);
  border-color:#ef4444;
  box-shadow:0 4px 20px rgba(239,68,68,.3);
}
.race-btn.play-main.is-playing:hover{
  box-shadow:0 6px 28px rgba(239,68,68,.4);
}
.race-btn.play-main.is-finished{
  background:linear-gradient(135deg,#22c55e,#16a34a);
  border-color:#22c55e;
  box-shadow:0 4px 20px rgba(34,197,94,.3);
}
.race-btn .ripple{
  position:absolute;border-radius:50%;
  background:rgba(255,255,255,.3);
  transform:scale(0);animation:rippleAnim .5s ease-out;
  pointer-events:none;
}
@keyframes rippleAnim{to{transform:scale(3);opacity:0}}

.speed-btns{display:flex;gap:4px}
.speed-btn{
  background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.1);
  color:#888;border-radius:8px;padding:8px 14px;cursor:pointer;
  font-size:.75rem;font-weight:700;
  transition:all .15s ease;
  outline:none;user-select:none;
}
.speed-btn:hover{background:rgba(255,255,255,.1);color:#ccc;transform:translateY(-1px)}
.speed-btn:active{transform:translateY(0) scale(.95);transition-duration:.05s}
.speed-btn:focus-visible{outline:2px solid #c084fc;outline-offset:2px}
.speed-btn.active{
  background:linear-gradient(135deg,#7c3aed,#6d28d9);
  color:white;border-color:#7c3aed;
  box-shadow:0 2px 10px rgba(124,58,237,.3);
}
.race-kbd-hint{
  font-size:.65rem;color:#555;text-align:center;
  padding:8px 0 0;letter-spacing:.3px;
}
.race-kbd-hint kbd{
  background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.1);
  border-radius:4px;padding:2px 6px;font-family:inherit;
  font-size:.6rem;color:#888;
}
.race-status{
  text-align:center;font-size:.75rem;color:#888;
  padding:6px 0;font-weight:600;
  transition:color .3s;
}
.race-status.complete{
  color:#22c55e;
  animation:statusGlow 1.5s ease-in-out infinite;
}
@keyframes statusGlow{
  0%,100%{text-shadow:none}
  50%{text-shadow:0 0 12px rgba(34,197,94,.4)}
}

/* ── Scout cards ────────────────────────────────────────── */
.scout-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:16px}
.scout-card{
  background:linear-gradient(145deg,rgba(20,20,43,.9),rgba(26,26,58,.9));
  backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);
  border:1px solid rgba(255,255,255,.07);border-radius:14px;
  padding:20px;position:relative;overflow:hidden;
  transition:transform .25s cubic-bezier(.4,0,.2,1),box-shadow .25s,border-color .25s;
}
.scout-card::before{
  content:'';position:absolute;top:0;left:0;right:0;height:3px;
  background:linear-gradient(90deg,transparent,rgba(124,58,237,.3),transparent);
  opacity:0;transition:opacity .3s;
}
.scout-card:hover{
  transform:translateY(-4px);
  box-shadow:0 12px 36px rgba(0,0,0,.5);
  border-color:rgba(124,58,237,.2);
}
.scout-card:hover::before{opacity:1}
.scout-card .rising-tag{
  position:absolute;top:12px;right:12px;
  background:linear-gradient(135deg,#22c55e,#16a34a);
  font-size:.6rem;font-weight:800;padding:3px 8px;border-radius:4px;
  color:white;letter-spacing:.5px;text-transform:uppercase;
  box-shadow:0 2px 8px rgba(34,197,94,.3);
}
.scout-card .golden-tag{
  position:absolute;top:12px;right:12px;
  background:linear-gradient(135deg,#f59e0b,#f97316);
  font-size:.6rem;font-weight:800;padding:3px 8px;border-radius:4px;
  color:#000;letter-spacing:.5px;text-transform:uppercase;
  box-shadow:0 2px 8px rgba(245,158,11,.3);
}
.scout-top{display:flex;align-items:center;gap:14px;margin-bottom:14px}
.scout-avatar{
  width:50px;height:50px;border-radius:50%;
  display:flex;align-items:center;justify-content:center;
  font-size:1.1rem;font-weight:800;color:white;flex-shrink:0;
  box-shadow:0 4px 12px rgba(0,0,0,.3);
  transition:transform .2s;
}
.scout-card:hover .scout-avatar{transform:scale(1.06)}
.scout-name{font-size:1rem;font-weight:700;color:#fff}
.scout-meta{font-size:.75rem;color:#888;margin-top:2px}
.scout-stats{
  display:grid;grid-template-columns:repeat(3,1fr);gap:8px;
  margin:12px 0;
}
.scout-stat{
  background:rgba(255,255,255,.04);border-radius:8px;
  padding:10px 8px;text-align:center;
  border:1px solid rgba(255,255,255,.03);
  transition:border-color .2s;
}
.scout-card:hover .scout-stat{border-color:rgba(255,255,255,.07)}
.scout-stat .val{font-size:1.2rem;font-weight:800;color:#fff}
.scout-stat .lbl{font-size:.6rem;color:#777;text-transform:uppercase;letter-spacing:.5px;margin-top:3px}
.breakout-track{
  width:100%;height:6px;background:rgba(255,255,255,.06);
  border-radius:3px;overflow:hidden;margin-top:12px;
}
.breakout-fill{height:100%;border-radius:3px;transition:width .8s ease}
.breakout-label{
  display:flex;justify-content:space-between;align-items:center;
  margin-top:6px;font-size:.7rem;color:#888;
}
.breakout-score{font-weight:800;font-size:.85rem}

/* ── Goal predictions ───────────────────────────────────── */
.goals-table{width:100%;border-collapse:collapse;font-size:.85rem}
.goals-table th{
  padding:10px 12px;text-align:left;font-size:.7rem;
  letter-spacing:1px;text-transform:uppercase;color:#555577;
  border-bottom:1px solid rgba(255,255,255,.07);
  position:sticky;top:0;background:#14142B;
}
.goals-table td{
  padding:8px 12px;border-bottom:1px solid rgba(255,255,255,.03);
  vertical-align:middle;transition:background .2s;
}
.goals-table tr:hover td{background:rgba(255,255,255,.04)}
.xg-bar-track{
  width:100%;height:8px;background:rgba(255,255,255,.06);
  border-radius:4px;overflow:hidden;
}
.xg-bar-fill{height:100%;border-radius:4px;transition:width .6s cubic-bezier(.4,0,.2,1)}

/* ── Betting Edge ───────────────────────────────────────── */
.league-filters{
  display:flex;gap:6px;flex-wrap:wrap;margin-bottom:20px;
}
.league-filter-btn{
  background:rgba(255,255,255,.05);
  border:1px solid rgba(255,255,255,.1);
  color:#999;border-radius:10px;padding:8px 16px;
  cursor:pointer;font-size:.75rem;font-weight:700;
  transition:all .2s;outline:none;user-select:none;
  display:flex;align-items:center;gap:5px;
}
.league-filter-btn:hover{background:rgba(255,255,255,.1);color:#ddd;transform:translateY(-1px)}
.league-filter-btn:active{transform:scale(.96)}
.league-filter-btn.active{
  background:var(--lf-color,#7c3aed);color:white;
  border-color:var(--lf-color,#7c3aed);
  box-shadow:0 2px 12px color-mix(in srgb,var(--lf-color,#7c3aed) 40%,transparent);
}
.league-filter-btn .lf-count{
  background:rgba(255,255,255,.15);border-radius:6px;
  padding:1px 6px;font-size:.65rem;
}
.betting-summary{
  display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
  gap:14px;margin-bottom:22px;
}
.bet-stat-card{
  background:rgba(20,20,43,.85);
  backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);
  border:1px solid rgba(255,255,255,.07);
  border-radius:14px;padding:18px;text-align:center;
  transition:border-color .3s,box-shadow .3s,transform .25s;
}
.bet-stat-card:hover{
  border-color:rgba(124,58,237,.2);
  box-shadow:0 8px 30px rgba(0,0,0,.4);
  transform:translateY(-3px);
}
.bet-stat-card .bsc-val{
  font-size:1.8rem;font-weight:900;line-height:1.1;
  margin-bottom:4px;
}
.bet-stat-card .bsc-lbl{
  font-size:.65rem;color:#888;text-transform:uppercase;
  letter-spacing:.7px;font-weight:600;
}
.bet-disclaimer{
  background:rgba(245,158,11,.06);
  border:1px solid rgba(245,158,11,.2);
  border-radius:10px;padding:12px 18px;
  font-size:.76rem;color:#d4a04a;line-height:1.5;
  margin-bottom:20px;display:flex;align-items:flex-start;gap:10px;
}
.bet-disclaimer svg{flex-shrink:0;margin-top:2px}

.edge-table-wrap{
  overflow-x:auto;border-radius:14px;
  border:1px solid rgba(255,255,255,.07);
  background:rgba(20,20,43,.85);
  backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);
}
.edge-table{width:100%;border-collapse:collapse;font-size:.78rem;min-width:1050px}
.edge-table th{
  padding:10px 12px;text-align:left;font-size:.62rem;
  letter-spacing:1.1px;text-transform:uppercase;color:#555577;
  border-bottom:1px solid rgba(255,255,255,.08);
  position:sticky;top:0;background:#14142B;z-index:1;
  white-space:nowrap;
}
.edge-table td{
  padding:8px 12px;border-bottom:1px solid rgba(255,255,255,.03);
  vertical-align:middle;transition:background .2s;
  white-space:nowrap;
}
.edge-table tr:hover td{background:rgba(124,58,237,.04)}
.edge-table tr[data-league]{display:table-row}
.edge-table tr.league-hidden{display:none}
.edge-table .team-cell{display:flex;align-items:center;gap:6px}

.league-badge{
  display:inline-flex;align-items:center;justify-content:center;
  padding:2px 7px;border-radius:4px;font-size:.58rem;
  font-weight:800;letter-spacing:.5px;color:white;
  white-space:nowrap;flex-shrink:0;
}

.edge-pill{
  display:inline-flex;align-items:center;gap:3px;
  padding:2px 8px;border-radius:5px;
  font-size:.7rem;font-weight:700;
}
.edge-pill.positive{background:rgba(34,197,94,.12);color:#22c55e}
.edge-pill.negative{background:rgba(239,68,68,.12);color:#ef4444}
.edge-pill.neutral{background:rgba(255,255,255,.06);color:#888}

.rec-badge{
  display:inline-block;padding:3px 9px;border-radius:5px;
  font-size:.64rem;font-weight:800;letter-spacing:.4px;
  text-transform:uppercase;
}
.rec-strong{background:linear-gradient(135deg,rgba(34,197,94,.2),rgba(34,197,94,.08));color:#22c55e;border:1px solid rgba(34,197,94,.25)}
.rec-value{background:rgba(59,130,246,.12);color:#60a5fa;border:1px solid rgba(59,130,246,.2)}
.rec-spec{background:rgba(245,158,11,.1);color:#f59e0b;border:1px solid rgba(245,158,11,.2)}
.rec-monitor{background:rgba(255,255,255,.05);color:#888;border:1px solid rgba(255,255,255,.08)}
.rec-fair{background:rgba(255,255,255,.04);color:#666;border:1px solid rgba(255,255,255,.06)}
.rec-avoid{background:rgba(239,68,68,.08);color:#ef4444;border:1px solid rgba(239,68,68,.15)}

.confidence-ring{
  width:34px;height:34px;border-radius:50%;
  display:inline-flex;align-items:center;justify-content:center;
  font-size:.68rem;font-weight:800;
  position:relative;
}
.confidence-ring::before{
  content:'';position:absolute;inset:0;border-radius:50%;
  border:3px solid rgba(255,255,255,.06);
}

.kelly-bar-track{
  width:70px;height:5px;background:rgba(255,255,255,.06);
  border-radius:3px;overflow:hidden;display:inline-block;
  vertical-align:middle;margin-right:5px;
}
.kelly-bar-fill{height:100%;border-radius:3px}

.ev-glow{text-shadow:0 0 8px currentColor;font-weight:800}

.bet-method-grid{
  display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
  gap:12px;margin-bottom:22px;
}
.bet-method{
  background:rgba(20,20,43,.7);
  border:1px solid rgba(255,255,255,.06);
  border-radius:12px;padding:14px;
  transition:border-color .3s;
}
.bet-method:hover{border-color:rgba(124,58,237,.15)}
.bet-method h4{
  font-size:.75rem;font-weight:700;color:#c084fc;
  margin-bottom:5px;display:flex;align-items:center;gap:5px;
}
.bet-method p{font-size:.68rem;color:#888;line-height:1.45}

.bayes-bar-track{
  width:60px;height:5px;background:rgba(255,255,255,.06);
  border-radius:3px;overflow:hidden;display:inline-block;
  vertical-align:middle;margin-right:5px;
}

/* ── Footer ─────────────────────────────────────────────── */
.site-footer{
  border-top:1px solid rgba(255,255,255,.05);
  padding:20px 36px;text-align:center;
  color:#444;font-size:.72rem;
}

/* ── Utility ────────────────────────────────────────────── */
.text-gradient{
  background:linear-gradient(90deg,#c084fc,#7c3aed);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
}
"""


# ---------------------------------------------------------------------------
# JavaScript
# ---------------------------------------------------------------------------

JAVASCRIPT = r"""
// ── Tab switching ────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    document.querySelectorAll('.tab-section').forEach(s => s.classList.remove('active'));
    const target = document.getElementById(btn.dataset.tab);
    if (target) target.classList.add('active');
  });
});

// ── Ripple effect for buttons ────────────────────────────
function addRipple(e) {
  const btn = e.currentTarget;
  const rect = btn.getBoundingClientRect();
  const ripple = document.createElement('span');
  ripple.className = 'ripple';
  const size = Math.max(rect.width, rect.height);
  ripple.style.width = ripple.style.height = size + 'px';
  ripple.style.left = (e.clientX - rect.left - size/2) + 'px';
  ripple.style.top  = (e.clientY - rect.top  - size/2) + 'px';
  btn.appendChild(ripple);
  ripple.addEventListener('animationend', () => ripple.remove());
}
document.querySelectorAll('.race-btn, .speed-btn').forEach(b => {
  b.addEventListener('click', addRipple);
});

// ── Bar-chart race ───────────────────────────────────────
(function() {
  const dataEl = document.getElementById('race-json');
  if (!dataEl) return;

  const data = JSON.parse(dataEl.textContent);
  const colorsEl = document.getElementById('race-colors-json');
  const logosEl  = document.getElementById('race-logos-json');
  const colors = colorsEl ? JSON.parse(colorsEl.textContent) : {};
  const logos  = logosEl  ? JSON.parse(logosEl.textContent)  : {};

  if (!data.length) return;

  const container = document.getElementById('race-bars');
  const teams = Object.keys(data[0].standings);
  const barH = 38;
  container.style.height = (teams.length * barH) + 'px';

  const slugify = s => s.replace(/\s+/g, '-').replace(/[^a-zA-Z0-9-]/g, '').toLowerCase();

  let prevRanks = {};
  teams.forEach((team, i) => { prevRanks[team] = i; });

  teams.forEach(team => {
    const bar = document.createElement('div');
    bar.className = 'race-bar';
    bar.id = 'rb-' + slugify(team);

    const logoUrl = logos[team] || '';
    const imgTag = logoUrl
      ? '<img src="' + logoUrl + '" onerror="this.style.display=\'none\'">'
      : '<div style="width:26px;height:26px;border-radius:50%;background:' + (colors[team]||'#555') + ';flex-shrink:0"></div>';

    const c = colors[team] || '#4A90E2';
    bar.innerHTML =
      '<span class="rpos">-</span>' +
      imgTag +
      '<span class="rname">' + team + '</span>' +
      '<div style="flex:1;height:22px;position:relative">' +
        '<div class="rfill" style="background:linear-gradient(90deg,' + c + ',' + c + '88);width:0%;position:absolute;top:0;left:0;height:100%;border-radius:5px;box-shadow:0 0 8px ' + c + '44"></div>' +
      '</div>' +
      '<span class="rpts">0</span>';
    container.appendChild(bar);
  });

  let frame = 0;
  let isPlaying = false;
  let speed = 700;
  let timer = null;
  let finished = false;

  function render(fi) {
    const d = data[fi];
    const sorted = Object.entries(d.standings)
      .sort((a,b) => b[1] - a[1] || a[0].localeCompare(b[0]));
    const maxPts = Math.max(...sorted.map(s => s[1]), 1);

    sorted.forEach(([team, pts], rank) => {
      const bar = document.getElementById('rb-' + slugify(team));
      if (!bar) return;
      bar.style.transform = 'translateY(' + (rank * barH) + 'px)';

      const fill = bar.querySelector('.rfill');
      fill.style.width = Math.max(2, (pts / maxPts) * 100) + '%';

      const ptsEl = bar.querySelector('.rpts');
      ptsEl.textContent = pts;

      const posEl = bar.querySelector('.rpos');
      const prev = prevRanks[team];

      const c = colors[team] || '#4A90E2';

      if (rank === 0) {
        fill.style.boxShadow = '0 0 20px ' + c + 'aa';
        bar.querySelector('.rname').style.color = '#fff';
        bar.querySelector('.rname').style.fontWeight = '800';
        ptsEl.style.color = c;
        ptsEl.style.transform = 'scale(1.1)';
        posEl.textContent = '\u2654';
        posEl.className = 'rpos leader';
      } else {
        fill.style.boxShadow = '0 0 6px ' + c + '33';
        bar.querySelector('.rname').style.color = '#ccc';
        bar.querySelector('.rname').style.fontWeight = '600';
        ptsEl.style.color = '#fff';
        ptsEl.style.transform = 'scale(1)';
        posEl.textContent = (rank + 1);

        if (fi > 0 && prev !== undefined) {
          if (rank < prev) {
            posEl.className = 'rpos up';
          } else if (rank > prev) {
            posEl.className = 'rpos down';
          } else {
            posEl.className = 'rpos';
          }
        } else {
          posEl.className = 'rpos';
        }
      }
    });

    sorted.forEach(([team], rank) => { prevRanks[team] = rank; });

    const mdEl = document.getElementById('matchday-num');
    const dtEl = document.getElementById('matchday-date');
    const prEl = document.getElementById('race-progress');
    const stEl = document.getElementById('race-status');
    if (mdEl) {
      mdEl.textContent = 'MW ' + d.matchday;
      mdEl.classList.remove('bump');
      void mdEl.offsetWidth;
      mdEl.classList.add('bump');
    }
    if (dtEl) dtEl.textContent = d.date;
    if (prEl) prEl.style.width = ((fi + 1) / data.length * 100) + '%';
    if (stEl) {
      if (fi >= data.length - 1) {
        stEl.textContent = 'Season complete';
        stEl.className = 'race-status complete';
      } else {
        stEl.textContent = 'MW ' + d.matchday + ' of ' + data[data.length-1].matchday;
        stEl.className = 'race-status';
      }
    }
  }

  function step() {
    if (frame >= data.length - 1) {
      pause();
      finished = true;
      updateUI();
      return;
    }
    frame++;
    finished = false;
    render(frame);
  }

  function play() {
    if (isPlaying) return;
    if (finished) { frame = 0; render(0); finished = false; }
    isPlaying = true;
    timer = setInterval(step, speed);
    updateUI();
  }

  function pause() {
    isPlaying = false;
    if (timer) { clearInterval(timer); timer = null; }
    updateUI();
  }

  function toggle() {
    if (isPlaying) pause(); else play();
  }

  function reset() {
    pause();
    frame = 0;
    finished = false;
    teams.forEach((team, i) => { prevRanks[team] = i; });
    render(0);
    updateUI();
  }

  function setSpeed(ms) {
    speed = ms;
    if (isPlaying) { clearInterval(timer); timer = setInterval(step, speed); }
    document.querySelectorAll('.speed-btn').forEach(b => {
      b.classList.toggle('active', parseInt(b.dataset.speed) === ms);
    });
  }

  function updateUI() {
    const pb = document.getElementById('play-btn');
    if (!pb) return;
    pb.classList.remove('is-playing', 'is-finished');
    if (finished) {
      pb.innerHTML = '\u21BA  Replay';
      pb.classList.add('is-finished');
    } else if (isPlaying) {
      pb.innerHTML = '\u23F8\uFE0F  Pause';
      pb.classList.add('is-playing');
    } else {
      pb.innerHTML = '\u25B6\uFE0F  Play';
    }
  }

  // Keyboard shortcuts
  document.addEventListener('keydown', function(e) {
    const raceTab = document.getElementById('sec-race');
    if (!raceTab || !raceTab.classList.contains('active')) return;
    if (e.target.tagName === 'INPUT') return;

    if (e.code === 'Space') { e.preventDefault(); toggle(); }
    else if (e.code === 'ArrowRight') { e.preventDefault(); if (!isPlaying) step(); }
    else if (e.code === 'ArrowLeft') {
      e.preventDefault();
      if (!isPlaying && frame > 0) { frame--; render(frame); }
    }
    else if (e.code === 'KeyR') { e.preventDefault(); reset(); }
  });

  window.racePlay  = play;
  window.racePause = pause;
  window.raceToggle = toggle;
  window.raceReset = reset;
  window.raceStep  = step;
  window.raceSetSpeed = setSpeed;

  render(0);
  updateUI();
})();

// ── League filter for betting table ─────────────────────
(function() {
  const btns = document.querySelectorAll('.league-filter-btn');
  if (!btns.length) return;

  btns.forEach(btn => {
    btn.addEventListener('click', () => {
      btns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const league = btn.dataset.league;
      const rows = document.querySelectorAll('#edge-table tbody tr[data-league]');
      rows.forEach(row => {
        if (league === 'ALL' || row.dataset.league === league) {
          row.classList.remove('league-hidden');
        } else {
          row.classList.add('league-hidden');
        }
      });
    });
  });
})();
"""


# ---------------------------------------------------------------------------
# HTML section builders
# ---------------------------------------------------------------------------

def _form_badges_html(form_str: str, max_show: int = 6) -> str:
    color_map = {"W": "#22c55e", "D": "#f59e0b", "L": "#ef4444"}
    badges = []
    visible = form_str[-max_show:] if len(form_str) > max_show else form_str
    for ch in visible:
        c = color_map.get(ch, "#555")
        badges.append(
            f'<span style="display:inline-flex;align-items:center;justify-content:center;'
            f'width:20px;height:20px;border-radius:4px;background:{c};'
            f'font-size:10px;font-weight:800;color:white;flex-shrink:0;">{ch}</span>'
        )
    return "".join(badges)


def _prob_bars_html(win_probs, cur_points, cur_gd, elo_ratings, form_strings):
    sorted_teams = sorted(win_probs.items(), key=lambda x: x[1], reverse=True)
    max_prob = max((p for _, p in sorted_teams), default=1e-6)
    rows = []
    for rank, (team, prob) in enumerate(sorted_teams):
        display = sname(team); logo = tlogo(team); color = tcolor(team)
        pct = prob * 100
        fill_pct = (prob / max_prob) * 80 if max_prob > 0 else 0
        is_leader = rank == 0
        is_contender = pct >= 0.5
        is_elim = pct < 0.01
        row_bg = ("rgba(255,215,0,.06)" if is_leader else
                  "rgba(255,255,255,.02)" if is_contender else "transparent")
        row_border = f"border-left:3px solid {color};" if not is_elim else "border-left:3px solid rgba(255,255,255,.1);"
        bar_color = color if not is_elim else "rgba(255,255,255,.12)"
        text_color = "white" if not is_elim else "#555"
        name_weight = "700" if is_leader else "500" if is_contender else "400"
        logo_html = (
            f'<img src="{logo}" width="36" height="36" style="object-fit:contain;flex-shrink:0" '
            f'onerror="this.style.display=\'none\'">' if logo else
            f'<div style="width:36px;height:36px;background:{color};border-radius:50%;flex-shrink:0"></div>'
        )
        form_html = _form_badges_html(form_strings.get(team, ""))
        pct_label = f"{pct:.1f}%" if pct >= 0.1 else (f"{pct:.2f}%" if pct > 0 else "0%")
        elim_badge = '<span style="font-size:10px;color:#555;margin-left:8px">Elim.</span>' if is_elim else ""
        rows.append(f"""
<div style="display:flex;align-items:center;gap:10px;padding:7px 16px 7px 12px;
            border-radius:8px;background:{row_bg};{row_border}margin-bottom:3px">
  {logo_html}
  <div style="min-width:130px;max-width:150px;flex-shrink:0">
    <div style="font-weight:{name_weight};font-size:.88rem;color:{text_color};white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{display}</div>
    <div style="display:flex;gap:3px;margin-top:4px">{form_html}</div>
  </div>
  <div style="flex:1;height:30px;background:rgba(255,255,255,.05);border-radius:6px;overflow:hidden;position:relative">
    <div style="height:100%;width:{fill_pct:.2f}%;background:{bar_color};border-radius:6px;min-width:{'4px' if not is_elim else '0px'};
                box-shadow:{'0 0 12px '+color+'66' if is_leader else 'none'};transition:width .8s ease"></div>
  </div>
  <div style="min-width:52px;text-align:right;font-size:.88rem;font-weight:700;color:{color if not is_elim else '#555'}">{pct_label}{elim_badge}</div>
</div>""")
    return "\n".join(rows)


def _standings_table_html(teams, cur_points, cur_gd, gf, elo_ratings,
                          win_probs, form_strings):
    ranked = sorted(teams, key=lambda t: (cur_points.get(t, 0), cur_gd.get(t, 0), gf.get(t, 0)), reverse=True)
    n = len(ranked)
    rows = []
    for pos, team in enumerate(ranked, 1):
        pts = cur_points.get(team, 0); gd_v = cur_gd.get(team, 0)
        elo_v = elo_ratings.get(team, 1500); prob = win_probs.get(team, 0) * 100
        form = form_strings.get(team, ""); logo = tlogo(team); color = tcolor(team)
        bg = ("rgba(255,215,0,.09)" if pos == 1 else
              "rgba(59,130,246,.08)" if pos <= 4 else
              "rgba(239,68,68,.08)" if pos >= n - 2 else "transparent")
        logo_html = (
            f'<img src="{logo}" width="22" height="22" '
            f'style="vertical-align:middle;margin-right:6px;object-fit:contain" '
            f'onerror="this.style.display=\'none\'">' if logo else ""
        )
        prob_bar = (
            f'<div style="display:inline-flex;align-items:center;gap:6px">'
            f'<div style="width:{min(prob*1.8,90):.0f}px;height:6px;background:{color};border-radius:3px;flex-shrink:0"></div>'
            f'<span style="font-size:.8rem">{prob:.1f}%</span></div>'
        )
        form_html = _form_badges_html(form, max_show=5)
        rows.append(
            f'<tr style="background:{bg}">'
            f'<td style="text-align:center;color:#777;font-size:.8rem">{pos}</td>'
            f'<td>{logo_html}<span style="font-size:.88rem">{sname(team)}</span></td>'
            f'<td style="text-align:center;font-weight:700">{pts}</td>'
            f'<td style="text-align:center;color:#999;font-size:.85rem">{gd_v:+d}</td>'
            f'<td style="text-align:center;color:#999;font-size:.8rem">{elo_v:.0f}</td>'
            f'<td><div style="display:flex;gap:3px">{form_html}</div></td>'
            f'<td>{prob_bar}</td></tr>'
        )
    return "\n".join(rows)


def _race_section_html(matchday_data_json: str, colors_json: str, logos_json: str) -> str:
    return f"""
<div class="card" style="margin-bottom:24px">
  <div class="card-header">
    Premier League Season Race
    <span style="font-size:.65rem;color:#888;text-transform:none;letter-spacing:0">
      Watch the table unfold matchday by matchday
    </span>
  </div>
  <div class="card-body">
    <div class="race-header">
      <div>
        <div class="race-matchday" id="matchday-num">MW 1</div>
        <div class="race-date" id="matchday-date"></div>
      </div>
    </div>
    <div class="race-progress-track">
      <div class="race-progress-fill" id="race-progress" style="width:0%"></div>
    </div>
    <div class="race-container" id="race-bars"></div>
    <div class="race-controls">
      <button class="race-btn" onclick="raceReset()">Reset</button>
      <button class="race-btn play-main" id="play-btn" onclick="raceToggle()">
        &#9654;&#65039;  Play
      </button>
      <button class="race-btn" onclick="raceStep()">Step &rarr;</button>
      <div style="width:1px;height:28px;background:rgba(255,255,255,.08);margin:0 8px"></div>
      <div class="speed-btns">
        <button class="speed-btn" data-speed="1000" onclick="raceSetSpeed(1000)">0.5x</button>
        <button class="speed-btn active" data-speed="700" onclick="raceSetSpeed(700)">1x</button>
        <button class="speed-btn" data-speed="350" onclick="raceSetSpeed(350)">2x</button>
        <button class="speed-btn" data-speed="150" onclick="raceSetSpeed(150)">4x</button>
      </div>
    </div>
    <div class="race-status" id="race-status"></div>
    <div class="race-kbd-hint">
      <kbd>Space</kbd> Play/Pause &nbsp;&middot;&nbsp;
      <kbd>&larr;</kbd><kbd>&rarr;</kbd> Step &nbsp;&middot;&nbsp;
      <kbd>R</kbd> Reset
    </div>
  </div>
</div>

<script type="application/json" id="race-json">{matchday_data_json}</script>
<script type="application/json" id="race-colors-json">{colors_json}</script>
<script type="application/json" id="race-logos-json">{logos_json}</script>
"""


def _scout_section_html(players_scored: list) -> str:
    rising = [p for p in players_scored if p["category"] == "rising_star"][:12]
    cards = []
    for p in rising:
        color = TEAM_COLORS.get(p["team"], DEFAULT_COLOR)
        logo = TEAM_LOGOS.get(p["team"], "")
        initials = "".join(w[0] for w in p["name"].split()[:2]).upper()
        pos_colors = {
            "ST": "#ef4444", "RW": "#f59e0b", "LW": "#f59e0b",
            "AM": "#8b5cf6", "CM": "#3b82f6", "CDM": "#3b82f6",
            "CB": "#22c55e", "LB": "#22c55e", "RB": "#22c55e",
            "GK": "#6b7280",
        }
        pos_color = pos_colors.get(p["pos"], "#6b7280")
        breakout = p.get("breakout", 50)
        br_color = ("#22c55e" if breakout >= 75 else
                    "#f59e0b" if breakout >= 55 else "#ef4444")
        logo_html = (
            f'<img src="{logo}" width="20" height="20" '
            f'style="object-fit:contain;vertical-align:middle;margin-right:4px" '
            f'onerror="this.style.display=\'none\'">' if logo else ""
        )
        cards.append(f"""
<div class="scout-card">
  <div class="rising-tag">Rising Star</div>
  <div class="scout-top">
    <div class="scout-avatar" style="background:linear-gradient(135deg,{color},{color}88)">
      {initials}
    </div>
    <div>
      <div class="scout-name">{p['name']}</div>
      <div class="scout-meta">
        {logo_html}{p['team']}
        &nbsp;&middot;&nbsp;
        <span style="background:{pos_color};padding:1px 6px;border-radius:3px;font-size:.65rem;color:white;font-weight:700">{p['pos']}</span>
        &nbsp;&middot;&nbsp; Age {p['age']}
        &nbsp;&middot;&nbsp; {p['nat']}
      </div>
    </div>
  </div>
  <div class="scout-stats">
    <div class="scout-stat"><div class="val">{p['goals']}</div><div class="lbl">Goals</div></div>
    <div class="scout-stat"><div class="val">{p['assists']}</div><div class="lbl">Assists</div></div>
    <div class="scout-stat"><div class="val">{p['apps']}</div><div class="lbl">Apps</div></div>
  </div>
  <div style="display:flex;justify-content:space-between;align-items:center;font-size:.75rem;color:#888;margin-bottom:4px">
    <span>Market Value</span>
    <span style="color:#fff;font-weight:700">&pound;{p['value_m']}m</span>
  </div>
  <div class="breakout-track">
    <div class="breakout-fill" style="width:{breakout}%;background:linear-gradient(90deg,{br_color},{br_color}88)"></div>
  </div>
  <div class="breakout-label">
    <span>Breakout Score</span>
    <span class="breakout-score" style="color:{br_color}">{breakout}</span>
  </div>
</div>""")

    return f"""
<div style="margin-bottom:24px">
  <h2 style="font-size:1.3rem;font-weight:800;margin-bottom:6px">
    <span class="text-gradient">Rising Stars</span>
  </h2>
  <p style="color:#888;font-size:.85rem;margin-bottom:20px">
    Young talent making waves in the Premier League — ranked by our breakout algorithm.
  </p>
  <div class="scout-grid">
    {"".join(cards)}
  </div>
</div>
"""


def _goals_section_html(player_predictions: list, cur_gf: dict) -> str:
    top = player_predictions[:15]
    if not top:
        return '<p style="color:#888">No goal prediction data available.</p>'

    max_total = max((p["total_predicted"] for p in top), default=1)
    rows = []
    for rank, p in enumerate(top, 1):
        color = TEAM_COLORS.get(p["team"], DEFAULT_COLOR)
        logo = TEAM_LOGOS.get(p["team"], "")
        logo_html = (
            f'<img src="{logo}" width="22" height="22" '
            f'style="object-fit:contain;vertical-align:middle;margin-right:6px" '
            f'onerror="this.style.display=\'none\'">' if logo else ""
        )
        bar_w = min(95, (p["total_predicted"] / max_total) * 95)
        trophy = ""
        if rank == 1:
            trophy = ' <span style="color:#f59e0b;font-size:.7rem">&#x1F3C6;</span>'
        elif rank <= 3:
            trophy = ' <span style="color:#f59e0b;font-size:.7rem">&#x1F947;</span>'

        rows.append(
            f'<tr>'
            f'<td style="text-align:center;color:#777;font-weight:700;font-size:.85rem">{rank}</td>'
            f'<td>{logo_html}<span style="font-weight:600">{p["name"]}</span>{trophy}</td>'
            f'<td style="text-align:center;color:#999;font-size:.8rem">{p["team"]}</td>'
            f'<td style="text-align:center;font-weight:700;color:#fff">{p["goals"]}</td>'
            f'<td style="text-align:center;color:#22c55e;font-weight:600">+{p["predicted_remaining"]}</td>'
            f'<td style="text-align:center;font-weight:800;color:{color};font-size:1rem">{p["total_predicted"]}</td>'
            f'<td style="width:30%">'
            f'<div class="xg-bar-track"><div class="xg-bar-fill" style="width:{bar_w:.1f}%;background:linear-gradient(90deg,{color},{color}88)"></div></div>'
            f'</td></tr>'
        )

    return f"""
<div class="card">
  <div class="card-header">
    Golden Boot Prediction
    <span style="font-size:.65rem;color:#888;text-transform:none;letter-spacing:0">
      Projected top scorers based on Dixon-Coles model
    </span>
  </div>
  <div style="overflow-x:auto">
    <table class="goals-table">
      <thead><tr>
        <th>#</th><th>Player</th><th>Club</th>
        <th>Current</th><th>Predicted</th><th>Total</th><th>Projection</th>
      </tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
  </div>
</div>
"""


def _betting_edge_section_html(opportunities: list, is_synthetic: bool = False) -> str:
    if not opportunities:
        return (
            '<div class="card"><div class="card-body" style="text-align:center;padding:40px">'
            '<p style="color:#888;font-size:1rem">No soccer betting markets found on Polymarket right now.</p>'
            '<p style="color:#555;font-size:.8rem;margin-top:8px">Check back closer to match days.</p>'
            '</div></div>'
        )

    value_bets = [o for o in opportunities if o["edge"] > 0]
    best_ev = max((o["ev"] for o in opportunities), default=0)
    avg_edge = sum(o["edge"] for o in value_bets) / len(value_bets) if value_bets else 0
    high_conf = sum(1 for o in opportunities if o["confidence"] >= 60)
    n_leagues = len(set(o["league_code"] for o in opportunities))

    best_ev_color = "#22c55e" if best_ev > 0 else "#ef4444"
    avg_edge_color = "#22c55e" if avg_edge > 0 else "#ef4444"

    synthetic_note = ""
    if is_synthetic:
        synthetic_note = (
            '<div style="background:rgba(124,58,237,.08);border:1px solid rgba(124,58,237,.2);'
            'border-radius:10px;padding:12px 18px;font-size:.76rem;color:#a78bfa;'
            'margin-bottom:18px;display:flex;align-items:center;gap:8px">'
            '<span style="font-size:1rem">&#x1F4CA;</span> '
            'Showing simulated markets based on Elo-Poisson model projections. '
            'Live Polymarket odds will appear here when available.'
            '</div>'
        )

    league_counts = {}
    for o in opportunities:
        lc = o["league_code"]
        league_counts[lc] = league_counts.get(lc, 0) + 1

    league_order = ["EPL", "UCL", "LAL", "BUN", "SEA", "MLS"]
    filter_btns = ['<button class="league-filter-btn active" data-league="ALL" style="--lf-color:#7c3aed">All <span class="lf-count">{}</span></button>'.format(len(opportunities))]
    for lc in league_order:
        if lc not in league_counts:
            continue
        cfg = LEAGUES.get(lc)
        if not cfg:
            continue
        filter_btns.append(
            f'<button class="league-filter-btn" data-league="{lc}" '
            f'style="--lf-color:{cfg.color}">'
            f'{cfg.icon} {cfg.name} <span class="lf-count">{league_counts[lc]}</span>'
            f'</button>'
        )
    filters_html = '<div class="league-filters">' + ''.join(filter_btns) + '</div>'

    methods_html = """
<div class="bet-method-grid">
  <div class="bet-method">
    <h4><span style="color:#22c55e">&#x2191;</span> Edge</h4>
    <p>Model probability minus market implied probability. Positive = market undervalues this outcome.</p>
  </div>
  <div class="bet-method">
    <h4><span style="color:#60a5fa">&#x1F4B0;</span> Expected Value</h4>
    <p>EV = (Model Prob &times; Payout) &minus; 1. Positive EV = expected profit long-term.</p>
  </div>
  <div class="bet-method">
    <h4><span style="color:#c084fc">&#x1F3AF;</span> Kelly Criterion</h4>
    <p>Variance-adjusted optimal bet sizing. Half-Kelly with uncertainty penalty. Capped at 20%.</p>
  </div>
  <div class="bet-method">
    <h4><span style="color:#f59e0b">&#x1F4CA;</span> Z-Score</h4>
    <p>Statistical significance of edge. |Z| &gt; 1.96 = 95% confidence the edge is real.</p>
  </div>
  <div class="bet-method">
    <h4><span style="color:#a78bfa">&#x1F52C;</span> Bayesian P(Edge)</h4>
    <p>Posterior probability that true value exceeds market price. Uses Beta distribution with model uncertainty.</p>
  </div>
  <div class="bet-method">
    <h4><span style="color:#22d3ee">&#x26A1;</span> Sharpe Ratio</h4>
    <p>Edge divided by model uncertainty. Higher = more reliable signal-to-noise ratio for the bet.</p>
  </div>
</div>
"""

    min_per_league = 8
    display_opps = []
    league_used = {}
    for o in opportunities:
        lc = o["league_code"]
        league_used.setdefault(lc, 0)
        if league_used[lc] < min_per_league:
            display_opps.append(o)
            league_used[lc] += 1
    remaining = [o for o in opportunities if o not in display_opps]
    slots_left = max(0, 80 - len(display_opps))
    display_opps.extend(remaining[:slots_left])
    display_opps.sort(key=lambda x: x["composite_score"], reverse=True)

    rows = []
    for o in display_opps:
        team = o["team"]
        color = o.get("team_color", DEFAULT_COLOR)
        lc = o["league_code"]
        league_cfg = LEAGUES.get(lc)
        league_color = o.get("league_color", "#555")

        league_badge = (
            f'<span class="league-badge" style="background:{league_color}">'
            f'{o.get("league_icon", "")} {lc}</span>'
        )

        team_html = (
            f'<div class="team-cell">'
            f'<div style="width:22px;height:22px;background:{color};border-radius:50%;flex-shrink:0"></div>'
            f'<span style="font-weight:600;color:#fff;font-size:.78rem">{team}</span>'
            f'</div>'
        )

        edge_cls = "positive" if o["edge"] > 0.005 else ("negative" if o["edge"] < -0.005 else "neutral")
        edge_arrow = "&#x25B2;" if o["edge"] > 0 else ("&#x25BC;" if o["edge"] < 0 else "&#x25CF;")
        edge_html = f'<span class="edge-pill {edge_cls}">{edge_arrow} {o["edge"]*100:+.1f}%</span>'

        ev_color = "#22c55e" if o["ev"] > 0 else "#ef4444"
        ev_cls = "ev-glow" if abs(o["ev"]) > 0.1 else ""
        ev_html = f'<span class="{ev_cls}" style="color:{ev_color};font-weight:700">{o["ev"]*100:+.1f}%</span>'

        kelly_pct = o["kelly"] * 100
        kelly_color = "#22c55e" if kelly_pct > 3 else ("#f59e0b" if kelly_pct > 0.5 else "#666")
        kelly_w = min(kelly_pct / 20 * 100, 100)
        kelly_html = (
            f'<div class="kelly-bar-track">'
            f'<div class="kelly-bar-fill" style="width:{kelly_w:.0f}%;background:{kelly_color}"></div>'
            f'</div>'
            f'<span style="color:{kelly_color};font-weight:700;font-size:.72rem">{kelly_pct:.1f}%</span>'
        )

        z = o["z_score"]
        z_color = "#22c55e" if abs(z) > 1.96 else ("#f59e0b" if abs(z) > 1.0 else "#888")
        z_html = f'<span style="color:{z_color};font-weight:700">{z:+.2f}</span>'

        p_edge = o.get("p_edge_real", 0)
        pe_color = "#22c55e" if p_edge > 0.7 else ("#f59e0b" if p_edge > 0.4 else "#888")
        pe_w = min(p_edge * 100, 100)
        pe_html = (
            f'<div class="bayes-bar-track">'
            f'<div class="kelly-bar-fill" style="width:{pe_w:.0f}%;background:{pe_color}"></div>'
            f'</div>'
            f'<span style="color:{pe_color};font-weight:700;font-size:.72rem">{p_edge*100:.0f}%</span>'
        )

        conf = o["confidence"]
        conf_color = ("#22c55e" if conf >= 70 else "#60a5fa" if conf >= 50 else
                      "#f59e0b" if conf >= 30 else "#ef4444")
        ring_shadow = f"0 0 10px {conf_color}44" if conf >= 60 else "none"
        conf_html = (
            f'<div class="confidence-ring" style="color:{conf_color};'
            f'box-shadow:{ring_shadow}">{conf}</div>'
        )

        rec = o["recommendation"]
        rec_cls_map = {
            "Strong Bet": "rec-strong", "Value Bet": "rec-value",
            "Speculative": "rec-spec", "Monitor": "rec-monitor",
            "Fair Price": "rec-fair", "Avoid": "rec-avoid",
        }
        rec_cls = rec_cls_map.get(rec, "rec-monitor")
        rec_html = f'<span class="rec-badge {rec_cls}">{rec}</span>'

        rows.append(
            f'<tr data-league="{lc}">'
            f'<td>{league_badge}</td>'
            f'<td>{team_html}</td>'
            f'<td style="text-align:center"><span style="color:#aaa">{o["market_prob"]*100:.1f}%</span></td>'
            f'<td style="text-align:center"><span style="color:{color};font-weight:700">{o["model_prob"]*100:.1f}%</span></td>'
            f'<td style="text-align:center">{edge_html}</td>'
            f'<td style="text-align:center">{ev_html}</td>'
            f'<td style="text-align:center">{kelly_html}</td>'
            f'<td style="text-align:center">{z_html}</td>'
            f'<td style="text-align:center">{pe_html}</td>'
            f'<td style="text-align:center">{conf_html}</td>'
            f'<td style="text-align:center">{rec_html}</td>'
            f'</tr>'
        )

    return f"""
{synthetic_note}

<div class="bet-disclaimer">
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#d4a04a" stroke-width="2">
    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
    <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
  </svg>
  <div>
    <strong>Disclaimer:</strong> This is a statistical analysis tool, not financial advice.
    All probabilities are model estimates with inherent uncertainty.
    Past performance does not predict future results. Gamble responsibly.
  </div>
</div>

<div class="betting-summary">
  <div class="bet-stat-card">
    <div class="bsc-val" style="color:#c084fc">{n_leagues}</div>
    <div class="bsc-lbl">Leagues Analyzed</div>
  </div>
  <div class="bet-stat-card">
    <div class="bsc-val" style="color:#22c55e">{len(value_bets)}</div>
    <div class="bsc-lbl">Value Bets Found</div>
  </div>
  <div class="bet-stat-card">
    <div class="bsc-val" style="color:{best_ev_color}">{best_ev*100:+.1f}%</div>
    <div class="bsc-lbl">Best Expected Value</div>
  </div>
  <div class="bet-stat-card">
    <div class="bsc-val" style="color:{avg_edge_color}">{avg_edge*100:+.1f}%</div>
    <div class="bsc-lbl">Avg Edge (Value Bets)</div>
  </div>
  <div class="bet-stat-card">
    <div class="bsc-val" style="color:#60a5fa">{high_conf}</div>
    <div class="bsc-lbl">High Confidence (&ge;60)</div>
  </div>
</div>

{filters_html}

{methods_html}

<div class="edge-table-wrap">
  <table class="edge-table" id="edge-table">
    <thead><tr>
      <th>League</th><th>Team</th><th>Market</th><th>Model</th>
      <th>Edge</th><th>EV</th>
      <th>Kelly</th><th>Z-Score</th><th>P(Edge)</th>
      <th>Conf</th><th>Signal</th>
    </tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
</div>
"""


# ---------------------------------------------------------------------------
# Elo chart
# ---------------------------------------------------------------------------

def build_elo_chart(elo_history: pd.DataFrame, elo_ratings: dict) -> str:
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
            mode="lines", name=sname(team),
            line=dict(color=color, width=2.5),
            hovertemplate=(
                f"<b>{sname(team)}</b><br>"
                "Date: %{x|%d %b %Y}<br>"
                "Elo: %{y:.0f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        paper_bgcolor=PANEL_BG, plot_bgcolor=PANEL_BG,
        font=dict(color="white", family="Inter, Segoe UI, Arial"),
        height=380, margin=dict(l=60, r=20, t=40, b=50),
        legend=dict(font=dict(color="white", size=10),
                    bgcolor="rgba(20,20,40,.8)",
                    bordercolor="rgba(255,255,255,.12)", borderwidth=1),
        xaxis=dict(gridcolor="rgba(255,255,255,.07)", tickfont=dict(color="#aaa")),
        yaxis=dict(title="Elo Rating", title_font=dict(color="#aaa", size=11),
                   gridcolor="rgba(255,255,255,.07)", tickfont=dict(color="#aaa")),
    )

    return pio.to_html(fig, full_html=False, include_plotlyjs="cdn",
                       config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# Full HTML assembly
# ---------------------------------------------------------------------------

def build_html(
    win_probs, cur_points, cur_gd, gf, elo_ratings,
    form_scores, form_strings, teams, target_season,
    remaining_count, is_complete, elo_chart_html,
    matchday_data, player_scores, player_predictions,
    betting_opportunities=None,
):
    winner = max(win_probs, key=win_probs.get)
    winner_prob = win_probs[winner] * 100
    w_color = tcolor(winner); w_logo = tlogo(winner); w_short = sname(winner)
    season_lbl = target_season.replace("-", "/")
    match_lbl = "Season complete" if is_complete else f"{remaining_count} fixtures remaining"

    logo_img = (
        f'<img src="{w_logo}" width="100" height="100" '
        f'style="filter:drop-shadow(0 0 18px {w_color});object-fit:contain" '
        f'onerror="this.style.display=\'none\'">'
        if w_logo else ""
    )

    by_prob = sorted(win_probs.items(), key=lambda x: x[1], reverse=True)
    runners_html = ""
    for t, p in by_prob[1:4]:
        c = tcolor(t); lg = tlogo(t)
        lgi = (f'<img src="{lg}" width="28" height="28" style="object-fit:contain" '
               f'onerror="this.style.display=\'none\'">' if lg else "")
        runners_html += (
            f'<div class="runner">{lgi}'
            f'<span style="color:{c};font-weight:700">{sname(t)}</span>'
            f'<span class="runner-pct">{p*100:.1f}%</span></div>'
        )

    prob_bars = _prob_bars_html(win_probs, cur_points, cur_gd, elo_ratings, form_strings)
    standings = _standings_table_html(teams, cur_points, cur_gd, gf, elo_ratings,
                                      win_probs, form_strings)

    # Prepare race data as JSON — map raw team names to display names
    race_data_display = []
    for snap in matchday_data:
        race_data_display.append({
            "matchday": snap["matchday"],
            "date": snap["date"],
            "standings": {sname(t): pts for t, pts in snap["standings"].items()},
        })
    colors_display = {sname(t): c for t, c in TEAM_COLORS.items()}
    logos_display = {sname(t): l for t, l in TEAM_LOGOS.items()}
    race_json = json.dumps(race_data_display)
    colors_json = json.dumps(colors_display)
    logos_json = json.dumps(logos_display)

    race_html = _race_section_html(race_json, colors_json, logos_json)
    scout_html = _scout_section_html(player_scores)
    goals_html = _goals_section_html(player_predictions, gf)

    is_synthetic = any(o.get("is_synthetic", False) for o in (betting_opportunities or []))
    betting_html = _betting_edge_section_html(betting_opportunities or [], is_synthetic)

    model_text = (
        "<strong>Model:</strong> Dixon-Coles time-weighted Poisson regression — "
        "estimates separate attack and defensive strength for every club using "
        "all historical results (exponential time decay so recent games count more), "
        "then simulates each remaining fixture by sampling goals from a Poisson "
        "distribution and adjusts for current-season form. "
        "10,000 Monte Carlo runs produce the title-win probabilities shown above."
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>PL Insights — Premier League {season_lbl} Analysis</title>
<meta name="description" content="Premier League {season_lbl} title race predictions, player scouting, and goal scorer analysis powered by Dixon-Coles Poisson modelling.">
<meta property="og:title" content="PL Insights — {season_lbl} Analysis Hub">
<meta property="og:description" content="Live title race predictions, animated season replay, rising star scouting, and Golden Boot projections.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">
<style>{CSS}</style>
</head>
<body>

<!-- ── Header ────────────────────────────────────────── -->
<header class="site-header">
  <div class="pl-badge">PL</div>
  <div class="header-text">
    <h1>PL Insights — {season_lbl}</h1>
    <p>Dixon-Coles Model &nbsp;&middot;&nbsp; {match_lbl} &nbsp;&middot;&nbsp; 10k simulations</p>
  </div>
</header>

<!-- ── Tab navigation ────────────────────────────────── -->
<nav class="tab-nav">
  <button class="tab-btn active" data-tab="sec-title">Title Race</button>
  <button class="tab-btn" data-tab="sec-race">Season Race</button>
  <button class="tab-btn" data-tab="sec-scout">Scout</button>
  <button class="tab-btn" data-tab="sec-goals">Top Scorers</button>
  <button class="tab-btn" data-tab="sec-betting">Betting Edge</button>
</nav>

<!-- ═══════════════════════════════════════════════════ -->
<!-- SECTION: TITLE RACE                                -->
<!-- ═══════════════════════════════════════════════════ -->
<div class="tab-section active" id="sec-title">
<div class="main">

  <div class="model-callout">{model_text}</div>

  <!-- Winner spotlight -->
  <div class="winner-card" style="
    background:linear-gradient(135deg,{CARD_BG} 0%,rgba({_hex_to_rgb(w_color)},.18) 100%);
    border:1px solid rgba({_hex_to_rgb(w_color)},.4);
    box-shadow:0 0 50px rgba({_hex_to_rgb(w_color)},.15);
  ">
    <div style="flex-shrink:0">{logo_img}</div>
    <div class="winner-info">
      <h2>Predicted Champion</h2>
      <div class="winner-name" style="color:{w_color};text-shadow:0 0 40px rgba({_hex_to_rgb(w_color)},.5)">{w_short}</div>
      <div class="winner-pct">
        <span style="font-size:1.6rem;font-weight:900;color:{w_color}">{winner_prob:.1f}%</span> title probability
      </div>
      <div class="runners">
        <span style="color:#666;font-size:.82rem;align-self:center">Also in contention:</span>
        {runners_html}
      </div>
    </div>
  </div>

  <!-- Two-column: probability bars + standings -->
  <div class="two-col">
    <div class="card">
      <div class="card-header">Title Win Probability</div>
      <div class="card-body" style="padding:12px 8px">{prob_bars}</div>
    </div>
    <div class="card">
      <div class="card-header">Current Standings</div>
      <div style="overflow-y:auto;max-height:640px">
        <table class="standings-tbl">
          <thead><tr><th>#</th><th>Club</th><th>Pts</th><th>GD</th><th>Elo</th><th>Form</th><th>Title %</th></tr></thead>
          <tbody>{standings}</tbody>
        </table>
      </div>
      <div class="legend">
        <span><span class="pill" style="background:rgba(255,215,0,.5)"></span>Leader</span>
        <span><span class="pill" style="background:rgba(59,130,246,.5)"></span>UCL spots</span>
        <span><span class="pill" style="background:rgba(239,68,68,.4)"></span>Relegation</span>
      </div>
    </div>
  </div>

  <div class="card">
    <div class="card-header">Elo Rating Trajectory — Top 8 (all seasons)</div>
    <div style="padding:4px">{elo_chart_html}</div>
  </div>

</div>
</div>

<!-- ═══════════════════════════════════════════════════ -->
<!-- SECTION: SEASON RACE                               -->
<!-- ═══════════════════════════════════════════════════ -->
<div class="tab-section" id="sec-race">
<div class="main">
  <h2 style="font-size:1.3rem;font-weight:800;margin-bottom:6px">
    <span class="text-gradient">Season Race — {season_lbl}</span>
  </h2>
  <p style="color:#888;font-size:.85rem;margin-bottom:20px">
    Watch the entire Premier League season unfold in real time.
    Hit <strong style="color:#c084fc">Play</strong> to animate, or step through matchday by matchday.
  </p>
  {race_html}
</div>
</div>

<!-- ═══════════════════════════════════════════════════ -->
<!-- SECTION: SCOUT                                     -->
<!-- ═══════════════════════════════════════════════════ -->
<div class="tab-section" id="sec-scout">
<div class="main">
  {scout_html}
</div>
</div>

<!-- ═══════════════════════════════════════════════════ -->
<!-- SECTION: TOP SCORERS                               -->
<!-- ═══════════════════════════════════════════════════ -->
<div class="tab-section" id="sec-goals">
<div class="main">
  <h2 style="font-size:1.3rem;font-weight:800;margin-bottom:6px">
    <span class="text-gradient">Goal Scorer Predictions</span>
  </h2>
  <p style="color:#888;font-size:.85rem;margin-bottom:20px">
    Projected top scorers based on the Dixon-Coles expected goals model and
    each player's current scoring share within their team.
  </p>
  {goals_html}
</div>
</div>

<!-- ═══════════════════════════════════════════════════ -->
<!-- SECTION: BETTING EDGE                              -->
<!-- ═══════════════════════════════════════════════════ -->
<div class="tab-section" id="sec-betting">
<div class="main">
  <h2 style="font-size:1.3rem;font-weight:800;margin-bottom:6px">
    <span class="text-gradient">Betting Edge — All Leagues</span>
  </h2>
  <p style="color:#888;font-size:.85rem;margin-bottom:20px">
    Multi-league edge analysis across EPL, Champions League, La Liga, Bundesliga, Serie A, and MLS.
    Elo-Poisson model with Bayesian edge detection, variance-adjusted Kelly, and ensemble probabilities.
  </p>
  {betting_html}
</div>
</div>

<!-- ── Footer ────────────────────────────────────────── -->
<footer class="site-footer">
  PL Insights &copy; 2025 &nbsp;&middot;&nbsp; Powered by Dixon-Coles Poisson Model
  &nbsp;&middot;&nbsp; Not affiliated with the Premier League
  <br>
  <span style="color:#555;font-size:.65rem">
    Data from openfootball &nbsp;&middot;&nbsp; Predictions are statistical estimates, not guarantees
  </span>
</footer>

<script>
{JAVASCRIPT}
</script>
</body>
</html>"""


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
            [sys.executable, os.path.join(os.path.dirname(__file__), "data_fetch.py")],
            check=True,
        )

    print("Loading match data...")
    completed, upcoming = load_season_data(data_path)

    if completed.empty:
        print("ERROR: No completed matches found.")
        return

    seasons_with_data = completed.groupby("season").size().sort_index().index.tolist()
    upcoming_seasons = upcoming["season"].unique().tolist() if not upcoming.empty else []
    all_seasons = sorted(set(seasons_with_data + upcoming_seasons))
    target_season = all_seasons[-1]
    if target_season not in seasons_with_data:
        target_season = seasons_with_data[-1]

    s_done = completed[completed["season"] == target_season]
    s_todo = (upcoming[upcoming["season"] == target_season]
              if not upcoming.empty else pd.DataFrame())

    remaining_count = len(s_todo)
    is_complete = remaining_count == 0

    print(f"Target season : {target_season}")
    print(f"Completed     : {len(s_done)} matches")
    print(f"Remaining     : {remaining_count} fixtures")

    # ── Core predictions ─────────────────────────────────────────────
    print("Computing form...")
    form_atk, form_def, form_strings = compute_goals_form(completed, target_season)

    print("Running Monte Carlo (50,000 iterations)...")
    (win_probs, top4_probs, rel_probs,
     cur_points, cur_gd, elo_ratings, teams,
     atk_home, atk_away, def_home, def_away,
     home_adv, rho, _fa, _fd) = run_monte_carlo(
        completed, upcoming, target_season
    )
    _, _, gf = compute_standings(completed, target_season)
    elo_history = compute_elo_history(completed)

    # ── Race animation data ──────────────────────────────────────────
    print("Computing matchday standings for race animation...")
    matchday_data = compute_matchday_standings(completed, target_season)

    # ── Player scouting ──────────────────────────────────────────────
    print("Scoring players for scouting section...")
    player_scores = compute_breakout_scores(PL_PLAYERS, elo_ratings, form_atk, sname)

    # ── Goal scorer predictions ──────────────────────────────────────
    print("Predicting top scorers...")
    team_xg = predict_remaining_goals(atk_home, atk_away, def_home, def_away,
                                      home_adv, upcoming, target_season,
                                      form_atk, form_def)

    # Map raw team names to display names for matching with player data
    display_gf = {sname(t): v for t, v in gf.items()}
    display_xg = {sname(t): v for t, v in team_xg.items()}

    player_predictions = []
    for p in player_scores:
        if p["goals"] == 0 and p["pos"] in ("CB", "GK", "LB", "RB", "CDM"):
            continue
        team = p["team"]
        remaining_xg = display_xg.get(team, 0)
        current_gf = display_gf.get(team, 1)
        goal_share = p["goals"] / max(current_gf, 1)
        predicted_remaining = remaining_xg * goal_share
        player_predictions.append({
            **p,
            "predicted_remaining": round(predicted_remaining, 1),
            "total_predicted": round(p["goals"] + predicted_remaining, 1),
        })
    player_predictions.sort(key=lambda x: x["total_predicted"], reverse=True)
    player_predictions = player_predictions[:20]

    # ── Polymarket betting edge (all soccer leagues) ────────────────
    print("Fetching Polymarket soccer markets (EPL, UCL, La Liga, Bundesliga, Serie A, MLS)...")
    raw_markets = fetch_all_soccer_markets()
    parsed = parse_markets(raw_markets)
    betting_opportunities = analyze_all_edges(
        parsed,
        epl_win_probs=win_probs,
        epl_elo_ratings=elo_ratings,
        epl_form_scores=form_atk,
        sname_fn=sname,
    )

    value_count = sum(1 for o in betting_opportunities if o["edge"] > 0)
    league_set = set(o["league_code"] for o in betting_opportunities)
    print(f"  Markets parsed: {len(parsed)}")
    print(f"  Leagues:        {', '.join(sorted(league_set))}")
    print(f"  Opportunities:  {len(betting_opportunities)} ({value_count} with positive edge)")
    if betting_opportunities:
        best = betting_opportunities[0]
        print(f"  Top pick:       {best['team']} ({best['league_name']}) — EV {best['ev']*100:+.1f}%, Conf {best['confidence']}")

    # ── Console summary ──────────────────────────────────────────────
    print("\nTitle-race probabilities:")
    for team, prob in sorted(win_probs.items(), key=lambda x: x[1], reverse=True):
        bar = "#" * int(prob * 40)
        print(f"  {sname(team):<22} {prob*100:6.2f}%  {bar}")

    # ── Build dashboard ──────────────────────────────────────────────
    print("\nBuilding Elo chart...")
    elo_chart_html = build_elo_chart(elo_history, elo_ratings)

    print("Assembling dashboard...")
    html = build_html(
        win_probs, cur_points, cur_gd, gf, elo_ratings,
        form_atk, form_strings, teams, target_season,
        remaining_count, is_complete, elo_chart_html,
        matchday_data, player_scores, player_predictions,
        betting_opportunities,
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
