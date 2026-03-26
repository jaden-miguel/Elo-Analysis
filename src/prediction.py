"""
Premier League Title-Race Prediction — Model v3
================================================
Major upgrades over v2:

1. SEPARATE HOME/AWAY PARAMETERS
   Each team gets four fitted parameters instead of two:
     atk_home  — attacking strength when playing at home
     atk_away  — attacking strength when playing away
     def_home  — defensive quality when playing at home
     def_away  — defensive quality when playing away
   Expected goals:
     xG_home = exp(home_adv + atk_home[h] - def_away[a])
     xG_away = exp(atk_away[a] - def_home[h])
   This captures the large empirical gap between a team's home and away
   performance that a single-parameter model cannot.

2. SEASON-BOOSTED TIME DECAY
   Current-season matches are weighted 2× above the base exponential decay,
   so the model tracks live squad quality without ignoring historical data.
   xi = 0.002 (half-weight at ~346 days), season_boost = 2.0.

3. GOALS-BASED FORM (replaces W/D/L form)
   Tracks goals scored/conceded over the last 8 matches with exponential
   intra-window weighting (newest = highest weight). Produces separate
   attack-form and defense-form multipliers applied to xG predictions.

4. VECTORISED MONTE CARLO  (50,000 simulations)
   All remaining fixtures are simulated in a single numpy call:
     rng.poisson(lam, size=(50_000, n_matches))
   Ranking and market probabilities computed with full numpy broadcasting
   — runs in < 3 seconds on a modern machine.

5. BETTER OPTIMISER INITIALISATION
   Attack/defense seed values are computed from team-level raw goal averages,
   giving the L-BFGS-B solver a warm start that converges in fewer iterations.

6. HEAD-TO-HEAD ADJUSTMENT
   For each upcoming fixture the last 6 direct meetings are inspected.
   A damped H2H factor (10% weight) adjusts the final xG, capturing
   tactical matchup tendencies not reflected in average-strength models.
"""

import json
import math
from collections import defaultdict
from datetime import date as dt_date

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIME_DECAY    = 0.002     # Exponential decay: half-weight at ~346 days
SEASON_BOOST  = 2.0       # Current-season weight multiplier
FORM_GAMES    = 8         # Matches used for form calculation
FORM_POWER    = 0.28      # Exponent applied to form ratios (damping factor)
H2H_GAMES     = 6         # Direct meetings used for H2H adjustment
H2H_WEIGHT    = 0.10      # How much H2H shifts the xG (10 %)
N_SIMS        = 50_000    # Monte Carlo simulations
N_GOAL_CAP    = 12        # Simulation goal cap
_EPS          = 1e-9      # Numerical floor


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_season_data(path: str = "data/matches.json"):
    with open(path) as f:
        raw = json.load(f)

    done, todo = [], []
    for m in raw:
        t1, t2 = m.get("team1"), m.get("team2")
        if not t1 or not t2:
            continue
        score_block = m.get("score") or {}
        ft = score_block.get("ft") if isinstance(score_block, dict) else None
        base = {"season": m.get("season"), "date": m.get("date"),
                "home_team": t1, "away_team": t2}
        if ft and len(ft) == 2 and ft[0] is not None and ft[1] is not None:
            try:
                done.append({**base, "home_goals": int(ft[0]), "away_goals": int(ft[1])})
            except (ValueError, TypeError):
                pass
        else:
            todo.append(base)

    _ec = ["season","date","home_team","away_team","home_goals","away_goals"]
    completed = pd.DataFrame(done) if done else pd.DataFrame(columns=_ec)
    upcoming  = pd.DataFrame(todo) if todo else pd.DataFrame(
        columns=["season","date","home_team","away_team"])

    if not completed.empty:
        completed["date"] = pd.to_datetime(completed["date"], errors="coerce")
        completed = completed.sort_values("date").reset_index(drop=True)

    return completed, upcoming


# ---------------------------------------------------------------------------
# Model fitting — Dixon-Coles v3 (home/away split)
# ---------------------------------------------------------------------------

def fit_dixoncoles(completed: pd.DataFrame,
                   time_decay: float = TIME_DECAY,
                   season_boost: float = SEASON_BOOST) -> tuple:
    """
    Fit a time-weighted Dixon-Coles Poisson model with separate home/away
    attack and defense parameters for each team.

    Parameter layout (4n + 2 total):
      params[0 : n]     atk_home[i]   — log attack strength at home
      params[n : 2n]    atk_away[i]   — log attack strength away
      params[2n: 3n]    def_home[i]   — log defensive quality at home
      params[3n: 4n]    def_away[i]   — log defensive quality away
      params[4n]        home_adv      — global home-field log advantage
      params[4n+1]      rho           — Dixon-Coles low-score correction

    Expected goals:
      xG_home = exp(home_adv + atk_home[h] - def_away[a])
      xG_away = exp(atk_away[a] - def_home[h])

    Returns
    -------
    atk_home, atk_away, def_home, def_away : dict[str, float]
    home_adv : float
    rho      : float
    teams    : list[str]
    """
    teams = sorted(set(
        completed["home_team"].tolist() + completed["away_team"].tolist()
    ))
    n   = len(teams)
    idx = {t: i for i, t in enumerate(teams)}

    # ── Time weights with season boost ──
    ref_date    = completed["date"].max()
    max_season  = completed["season"].max()
    days_ago    = (ref_date - completed["date"]).dt.days.fillna(730).clip(0, 1460).values
    base_weight = np.exp(-time_decay * days_ago)
    season_mult = np.where(completed["season"] == max_season, season_boost, 1.0)
    weights     = base_weight * season_mult
    # Renormalise so effective sample size ~ n_matches
    weights     = weights / weights.mean()

    hi = np.array([idx[t] for t in completed["home_team"]])
    ai = np.array([idx[t] for t in completed["away_team"]])
    hg = completed["home_goals"].values.astype(np.int32)
    ag = completed["away_goals"].values.astype(np.int32)

    # ── Warm-start initialisation from raw goal averages ──
    avg_hs = completed.groupby("home_team")["home_goals"].mean()
    avg_as = completed.groupby("away_team")["away_goals"].mean()
    avg_hc = completed.groupby("home_team")["away_goals"].mean()
    avg_ac = completed.groupby("away_team")["home_goals"].mean()
    mu_hs  = avg_hs.mean();   mu_as  = avg_as.mean()
    mu_hc  = avg_hc.mean();   mu_ac  = avg_ac.mean()

    x0 = np.zeros(4 * n + 2)
    for i, team in enumerate(teams):
        x0[i]         = np.log(max(avg_hs.get(team, mu_hs), 0.3) / max(mu_hs, 0.3)) * 0.6
        x0[n + i]     = np.log(max(avg_as.get(team, mu_as), 0.3) / max(mu_as, 0.3)) * 0.6
        x0[2*n + i]   = np.log(max(mu_hc, 0.3) / max(avg_hc.get(team, mu_hc), 0.3)) * 0.6
        x0[3*n + i]   = np.log(max(mu_ac, 0.3) / max(avg_ac.get(team, mu_ac), 0.3)) * 0.6
    x0[4*n]     = 0.22
    x0[4*n + 1] = -0.08

    # ── Negative log-likelihood (vectorised) ──
    def neg_ll(params: np.ndarray) -> float:
        ah = params[:n];      aa = params[n:2*n]
        dh = params[2*n:3*n]; da = params[3*n:4*n]
        hadv = params[4*n]
        rho  = float(np.clip(params[4*n+1], -0.4, 0.0))

        lam = np.exp(hadv + ah[hi] - da[ai])
        mu  = np.exp(aa[ai] - dh[hi])
        lam = np.maximum(lam, _EPS)
        mu  = np.maximum(mu,  _EPS)

        tau = np.ones(len(lam))
        m00 = (hg==0)&(ag==0); m01 = (hg==0)&(ag==1)
        m10 = (hg==1)&(ag==0); m11 = (hg==1)&(ag==1)
        tau[m00] = np.maximum(1.0 - lam[m00]*mu[m00]*rho, _EPS)
        tau[m01] = 1.0 + lam[m01]*rho
        tau[m10] = 1.0 + mu[m10]*rho
        tau[m11] = 1.0 - rho

        ll_h = hg * np.log(lam) - lam - gammaln(hg + 1)
        ll_a = ag * np.log(mu)  - mu  - gammaln(ag + 1)
        ll   = weights * (np.log(np.maximum(tau, _EPS)) + ll_h + ll_a)

        # Separate L2 penalties: attack can deviate more than defense
        reg = 2e-4 * np.sum(params[:2*n]**2) + 3e-4 * np.sum(params[2*n:4*n]**2)
        return -(ll.sum()) + reg

    res = minimize(neg_ll, x0, method="L-BFGS-B",
                   options={"maxiter": 3000, "ftol": 1e-13, "gtol": 1e-9,
                            "maxfun": 60000})

    p = res.x
    atk_home = {teams[i]: float(p[i])         for i in range(n)}
    atk_away = {teams[i]: float(p[n + i])     for i in range(n)}
    def_home = {teams[i]: float(p[2*n + i])   for i in range(n)}
    def_away = {teams[i]: float(p[3*n + i])   for i in range(n)}
    home_adv = float(p[4*n])
    rho      = float(np.clip(p[4*n + 1], -0.4, 0.0))
    return atk_home, atk_away, def_home, def_away, home_adv, rho, teams


# ---------------------------------------------------------------------------
# Goals-based form (replaces W/D/L form)
# ---------------------------------------------------------------------------

def compute_goals_form(completed: pd.DataFrame, season: str,
                        n_games: int = FORM_GAMES) -> tuple[dict, dict, dict]:
    """
    Compute goals-based form multipliers for the last n_games in target season.

    Returns
    -------
    form_atk     : {team: float}  ratio of recent goals-scored to season avg  (>1 = hot attack)
    form_def     : {team: float}  ratio of season avg conceded to recent conceded  (>1 = solid defense)
    form_strings : {team: str}    'WWDLW' string for display  (newest on right)
    """
    df = (completed[completed["season"] == season]
          .sort_values("date")
          .copy())

    all_goals = df["home_goals"].tolist() + df["away_goals"].tolist()
    lg_avg    = float(np.mean(all_goals)) if all_goals else 1.2

    all_teams = set(df["home_team"].tolist()) | set(df["away_team"].tolist())

    form_atk:     dict[str, float] = {}
    form_def:     dict[str, float] = {}
    form_strings: dict[str, str]   = {}

    for team in all_teams:
        mask   = (df["home_team"] == team) | (df["away_team"] == team)
        recent = df[mask].tail(n_games)

        gf, ga, wdl = [], [], []
        for _, row in recent.iterrows():
            is_home = (row["home_team"] == team)
            scored   = int(row["home_goals"]) if is_home else int(row["away_goals"])
            conceded = int(row["away_goals"]) if is_home else int(row["home_goals"])
            gf.append(float(scored))
            ga.append(float(conceded))
            wdl.append("W" if scored > conceded else ("L" if scored < conceded else "D"))

        if gf:
            # Exponential intra-window weighting (newest highest)
            w = np.exp(np.linspace(-1.0, 0.0, len(gf)))
            gf_avg = float(np.average(gf, weights=w))
            ga_avg = float(np.average(ga, weights=w))
            form_atk[team]  = float(np.clip(gf_avg / max(lg_avg, 0.3), 0.35, 2.8))
            form_def[team]  = float(np.clip(lg_avg  / max(ga_avg, 0.3), 0.35, 2.8))
        else:
            form_atk[team] = 1.0
            form_def[team] = 1.0

        form_strings[team] = "".join(wdl)

    return form_atk, form_def, form_strings


# ---------------------------------------------------------------------------
# Head-to-head adjustment
# ---------------------------------------------------------------------------

def compute_h2h_factor(completed: pd.DataFrame,
                        home_team: str, away_team: str,
                        n: int = H2H_GAMES,
                        weight: float = H2H_WEIGHT) -> tuple[float, float]:
    """
    Compute head-to-head goal adjustment factors for a specific fixture.

    Returns
    -------
    h2h_home_mult, h2h_away_mult : float
        Multipliers to apply on top of model xG.
        1.0 = no H2H effect.  Requires >= 3 meetings to activate.
    """
    h2h = completed[
        ((completed["home_team"] == home_team) & (completed["away_team"] == away_team)) |
        ((completed["home_team"] == away_team) & (completed["away_team"] == home_team))
    ].sort_values("date").tail(n)

    if len(h2h) < 3:
        return 1.0, 1.0

    total_h_goals, total_a_goals = 0.0, 0.0
    for _, row in h2h.iterrows():
        if row["home_team"] == home_team:
            total_h_goals += row["home_goals"]
            total_a_goals += row["away_goals"]
        else:
            total_h_goals += row["away_goals"]
            total_a_goals += row["home_goals"]

    avg_h = total_h_goals / len(h2h)
    avg_a = total_a_goals / len(h2h)
    all_avg = (avg_h + avg_a) / 2.0 if (avg_h + avg_a) > 0 else 1.2

    # H2H goal ratio relative to expected equal split
    ratio_h = float(np.clip(avg_h / max(all_avg, 0.3), 0.5, 2.0))
    ratio_a = float(np.clip(avg_a / max(all_avg, 0.3), 0.5, 2.0))

    # Damped multiplier: blend towards 1.0 using H2H_WEIGHT
    mult_h = 1.0 + weight * (ratio_h - 1.0)
    mult_a = 1.0 + weight * (ratio_a - 1.0)
    return float(np.clip(mult_h, 0.8, 1.2)), float(np.clip(mult_a, 0.8, 1.2))


# ---------------------------------------------------------------------------
# Current-season standings
# ---------------------------------------------------------------------------

def compute_standings(completed: pd.DataFrame,
                      season: str) -> tuple[dict, dict, dict]:
    pts: dict[str, int] = defaultdict(int)
    gd:  dict[str, int] = defaultdict(int)
    gf:  dict[str, int] = defaultdict(int)
    for _, row in completed[completed["season"] == season].iterrows():
        hg, ag = int(row["home_goals"]), int(row["away_goals"])
        h,  a  = row["home_team"],       row["away_team"]
        if hg > ag:   pts[h] += 3
        elif hg < ag: pts[a] += 3
        else:         pts[h] += 1; pts[a] += 1
        gd[h] += hg-ag; gd[a] += ag-hg
        gf[h] += hg;    gf[a] += ag
    return dict(pts), dict(gd), dict(gf)


# ---------------------------------------------------------------------------
# Elo history (display only)
# ---------------------------------------------------------------------------

def compute_elo_history(completed: pd.DataFrame,
                         k: int = 32, home_adv: float = 50.0) -> pd.DataFrame:
    START = 1500.0
    ratings: dict[str, float] = defaultdict(lambda: START)
    rows = []
    for _, row in completed.iterrows():
        h, a = row["home_team"], row["away_team"]
        rh, ra = ratings[h], ratings[a]
        sh = (1.0 if row["home_goals"]>row["away_goals"]
              else 0.0 if row["home_goals"]<row["away_goals"] else 0.5)
        eh = 1.0 / (1.0 + 10.0**((ra - rh - home_adv)/400.0))
        ratings[h] = rh + k*(sh - eh)
        ratings[a] = ra + k*((1-sh) - (1-eh))
        rows.append({"date":row["date"],"season":row["season"],"team":h,"rating":ratings[h]})
        rows.append({"date":row["date"],"season":row["season"],"team":a,"rating":ratings[a]})
    return pd.DataFrame(rows)


def _simple_elo(completed: pd.DataFrame, k: int = 32) -> dict:
    START = 1500.0
    ratings: dict[str, float] = defaultdict(lambda: START)
    for _, row in completed.iterrows():
        h, a = row["home_team"], row["away_team"]
        rh, ra = ratings[h], ratings[a]
        sh = (1.0 if row["home_goals"]>row["away_goals"]
              else 0.0 if row["home_goals"]<row["away_goals"] else 0.5)
        eh = 1.0/(1.0+10.0**((ra-rh-50)/400.0))
        ratings[h] = rh + k*(sh-eh)
        ratings[a] = ra + k*((1-sh)-(1-eh))
    return dict(ratings)


# ---------------------------------------------------------------------------
# DC tau helper
# ---------------------------------------------------------------------------

def _dc_tau(hg: int, ag: int, lam: float, mu: float, rho: float) -> float:
    if   hg==0 and ag==0: return max(1.0 - lam*mu*rho, _EPS)
    elif hg==0 and ag==1: return 1.0 + lam*rho
    elif hg==1 and ag==0: return 1.0 + mu*rho
    elif hg==1 and ag==1: return 1.0 - rho
    return 1.0


# ---------------------------------------------------------------------------
# Analytical match probabilities (for betting guide)
# ---------------------------------------------------------------------------

def _xg(home_team, away_team,
        atk_home, atk_away, def_home, def_away,
        home_adv, form_atk, form_def,
        completed=None) -> tuple[float, float]:
    """Compute final expected goals for a fixture including all adjustments."""
    DEFAULT_A = float(np.mean(list(atk_home.values()))) if atk_home else 0.0
    DEFAULT_D = float(np.mean(list(def_home.values()))) if def_home else 0.0

    lam = math.exp(home_adv
                   + atk_home.get(home_team, DEFAULT_A)
                   - def_away.get(away_team, DEFAULT_D))
    mu  = math.exp(atk_away.get(away_team, DEFAULT_A)
                   - def_home.get(home_team, DEFAULT_D))

    # Goals-based form adjustment
    fa_h = form_atk.get(home_team, 1.0)
    fd_h = form_def.get(home_team, 1.0)
    fa_a = form_atk.get(away_team, 1.0)
    fd_a = form_def.get(away_team, 1.0)
    lam *= (fa_h ** FORM_POWER) * (fd_a ** FORM_POWER)
    mu  *= (fa_a ** FORM_POWER) * (fd_h ** FORM_POWER)

    # H2H adjustment
    if completed is not None:
        h2h_h, h2h_a = compute_h2h_factor(completed, home_team, away_team)
        lam *= h2h_h
        mu  *= h2h_a

    return max(0.05, lam), max(0.05, mu)


def compute_match_probabilities(
    home_team: str, away_team: str,
    atk_home: dict, atk_away: dict,
    def_home: dict, def_away: dict,
    home_adv: float, rho: float,
    form_atk: dict, form_def: dict,
    completed: pd.DataFrame = None,
    max_goals: int = 11,
) -> dict:
    """
    Analytically compute full probability breakdown for a single fixture.
    Includes H2H adjustment when completed data is provided.
    """
    from scipy.stats import poisson as sp

    xg_h, xg_a = _xg(home_team, away_team,
                      atk_home, atk_away, def_home, def_away,
                      home_adv, form_atk, form_def, completed)

    joint = np.zeros((max_goals+1, max_goals+1))
    for hg in range(max_goals+1):
        ph = sp.pmf(hg, xg_h)
        for ag in range(max_goals+1):
            tau = _dc_tau(hg, ag, xg_h, xg_a, rho)
            joint[hg, ag] = tau * ph * sp.pmf(ag, xg_a)

    joint = np.maximum(joint, 0.0)
    s = joint.sum()
    if s > 0: joint /= s

    home_win = float(sum(joint[h,a] for h in range(max_goals+1)
                         for a in range(max_goals+1) if h > a))
    draw     = float(sum(joint[k,k] for k in range(max_goals+1)))
    away_win = float(max(0.0, 1.0 - home_win - draw))

    totals = np.zeros(2*max_goals+2)
    for h in range(max_goals+1):
        for a in range(max_goals+1):
            totals[h+a] += joint[h,a]
    cum     = np.cumsum(totals)
    over_15 = float(max(0.0, 1.0 - cum[1]))
    over_25 = float(max(0.0, 1.0 - cum[2]))
    over_35 = float(max(0.0, 1.0 - cum[3]))

    p_h0 = float(joint[0,:].sum())
    p_a0 = float(joint[:,0].sum())
    btts = float(max(0.0, min(1.0, 1.0 - p_h0 - p_a0 + joint[0,0])))

    # Correct score (most likely scoreline)
    flat_idx     = np.argmax(joint)
    best_h, best_a = divmod(flat_idx, max_goals+1)
    cs_prob      = float(joint[best_h, best_a])

    return {
        "home_win": round(max(0.0,min(1.0,home_win)), 4),
        "draw":     round(max(0.0,min(1.0,draw)),     4),
        "away_win": round(max(0.0,min(1.0,away_win)), 4),
        "xg_home":  round(xg_h, 3),
        "xg_away":  round(xg_a, 3),
        "over_15":  round(over_15, 4),
        "over_25":  round(over_25, 4),
        "over_35":  round(over_35, 4),
        "btts":     round(btts, 4),
        "cs_home":  int(best_h),
        "cs_away":  int(best_a),
        "cs_prob":  round(cs_prob, 4),
    }


def predict_upcoming_fixtures(
    upcoming: pd.DataFrame, completed: pd.DataFrame,
    target_season: str,
    atk_home: dict, atk_away: dict,
    def_home: dict, def_away: dict,
    home_adv: float, rho: float,
    form_atk: dict, form_def: dict,
    n: int = 10, from_date: str = None,
) -> list[dict]:
    rem = (upcoming[upcoming["season"] == target_season]
           .copy().sort_values("date"))
    if from_date:
        rem = rem[rem["date"] >= from_date]
    rem = rem.head(n)

    results = []
    for _, row in rem.iterrows():
        h, a = row["home_team"], row["away_team"]
        probs = compute_match_probabilities(
            h, a, atk_home, atk_away, def_home, def_away,
            home_adv, rho, form_atk, form_def, completed
        )
        results.append({"date": str(row["date"])[:10],
                         "home_team": h, "away_team": a, **probs})
    return results


# ---------------------------------------------------------------------------
# Vectorised Monte Carlo — 50,000 simulations
# ---------------------------------------------------------------------------

def run_monte_carlo(
    completed: pd.DataFrame,
    upcoming:  pd.DataFrame,
    target_season: str,
    n_simulations: int = N_SIMS,
) -> tuple:
    """
    Fit the v3 model and simulate the remaining season.

    Returns a 14-element tuple (all consumed by predict_dashboard.py):
      win_probs, top4_probs, rel_probs,
      cur_points, cur_gd, elo_ratings, teams,
      atk_home, atk_away, def_home, def_away, home_adv, rho,
      form_atk, form_def
    """
    print("  Fitting Dixon-Coles v3 (home/away split)...", end=" ", flush=True)
    atk_home, atk_away, def_home, def_away, home_adv, rho, all_teams = \
        fit_dixoncoles(completed)
    print("done")

    cur_points, cur_gd, _ = compute_standings(completed, target_season)
    form_atk, form_def, _ = compute_goals_form(completed, target_season)

    remaining = upcoming[upcoming["season"] == target_season].copy()

    # All teams in the season
    s_done = completed[completed["season"] == target_season]
    s_todo = remaining
    season_teams: set[str] = set()
    for df in (s_done, s_todo):
        for col in ("home_team", "away_team"):
            if col in df.columns:
                season_teams.update(df[col].dropna())
    for t in season_teams:
        cur_points.setdefault(t, 0)

    teams_list = sorted(season_teams)
    n_teams    = len(teams_list)
    team_idx   = {t: i for i, t in enumerate(teams_list)}

    elo_ratings = _simple_elo(completed)

    DEFAULT_AH = float(np.mean(list(atk_home.values()))) if atk_home else 0.0
    DEFAULT_AA = float(np.mean(list(atk_away.values()))) if atk_away else 0.0
    DEFAULT_DH = float(np.mean(list(def_home.values()))) if def_home else 0.0
    DEFAULT_DA = float(np.mean(list(def_away.values()))) if def_away else 0.0

    # ── Season already finished ──
    if remaining.empty:
        winner = max(season_teams,
                     key=lambda t: (cur_points.get(t,0), cur_gd.get(t,0)))
        wp = {t: 0.0 for t in season_teams}; wp[winner] = 1.0
        z  = {t: 0.0 for t in season_teams}
        return (wp, z, z, cur_points, cur_gd, elo_ratings, teams_list,
                atk_home, atk_away, def_home, def_away, home_adv, rho,
                form_atk, form_def)

    # ── Pre-compute expected goals for all remaining fixtures ──
    n_matches     = len(remaining)
    home_teams_r  = remaining["home_team"].values
    away_teams_r  = remaining["away_team"].values
    xg_home_arr   = np.empty(n_matches)
    xg_away_arr   = np.empty(n_matches)

    for m, (h, a) in enumerate(zip(home_teams_r, away_teams_r)):
        lam_base = math.exp(home_adv
                            + atk_home.get(h, DEFAULT_AH)
                            - def_away.get(a, DEFAULT_DA))
        mu_base  = math.exp(atk_away.get(a, DEFAULT_AA)
                            - def_home.get(h, DEFAULT_DH))
        # Form (separate attack/defense ratios)
        fa_h = form_atk.get(h, 1.0); fd_h = form_def.get(h, 1.0)
        fa_a = form_atk.get(a, 1.0); fd_a = form_def.get(a, 1.0)
        lam_base *= (fa_h ** FORM_POWER) * (fd_a ** FORM_POWER)
        mu_base  *= (fa_a ** FORM_POWER) * (fd_h ** FORM_POWER)
        # H2H
        h2h_h, h2h_a = compute_h2h_factor(completed, h, a)
        xg_home_arr[m] = max(0.05, lam_base * h2h_h)
        xg_away_arr[m] = max(0.05, mu_base  * h2h_a)

    # ── Vectorised simulation ──
    print(f"  Simulating {n_simulations:,} seasons...", end=" ", flush=True)
    rng = np.random.default_rng(42)

    # Sample goals: shape (n_sim, n_matches)
    hg_all = rng.poisson(xg_home_arr, size=(n_simulations, n_matches))
    ag_all = rng.poisson(xg_away_arr, size=(n_simulations, n_matches))

    # Points per match
    home_pts = np.where(hg_all > ag_all, 3,
               np.where(hg_all == ag_all, 1, 0))   # (n_sim, n_match)
    away_pts = np.where(ag_all > hg_all, 3,
               np.where(ag_all == hg_all, 1, 0))

    # Home/away team indices
    h_idx = np.array([team_idx.get(t, 0) for t in home_teams_r])
    a_idx = np.array([team_idx.get(t, 0) for t in away_teams_r])

    # Accumulate simulated points: (n_sim, n_teams)
    sim_pts = np.zeros((n_simulations, n_teams), dtype=np.int32)
    for i, t in enumerate(teams_list):
        sim_pts[:, i] = cur_points.get(t, 0)
    for m in range(n_matches):
        sim_pts[:, h_idx[m]] += home_pts[:, m]
        sim_pts[:, a_idx[m]] += away_pts[:, m]

    # Add GD as tie-breaker (tiny fractional component)
    gd_arr   = np.array([cur_gd.get(t, 0) for t in teams_list], dtype=float)
    sort_key = sim_pts.astype(float) + gd_arr[None, :] * 1e-4
    ranked   = np.argsort(-sort_key, axis=1)           # (n_sim, n_teams) ascending rank

    # Win / Top-4 / Relegation counts
    wins_idx = ranked[:, 0]
    t4_idx   = ranked[:, :4].ravel()
    rel_idx  = ranked[:, max(0, n_teams-3):].ravel()

    win_counts  = np.bincount(wins_idx, minlength=n_teams)
    top4_counts = np.bincount(t4_idx,   minlength=n_teams)
    rel_counts  = np.bincount(rel_idx,  minlength=n_teams)

    print("done")

    # Each team appears at most once per simulation in each bucket,
    # so the denominator is always n_simulations.
    win_probs  = {teams_list[i]: win_counts[i]  / n_simulations for i in range(n_teams)}
    top4_probs = {teams_list[i]: top4_counts[i] / n_simulations for i in range(n_teams)}
    rel_probs  = {teams_list[i]: rel_counts[i]  / n_simulations for i in range(n_teams)}

    return (win_probs, top4_probs, rel_probs, cur_points, cur_gd, elo_ratings, teams_list,
            atk_home, atk_away, def_home, def_away, home_adv, rho, form_atk, form_def)
