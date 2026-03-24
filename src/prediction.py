"""
Premier League Title-Race Prediction — Advanced Model
======================================================
Uses the Dixon-Coles (1997) time-weighted Poisson model to estimate each
team's attack and defensive strength, then runs a full Monte Carlo simulation
of remaining fixtures to produce championship win probabilities.

Why this is more accurate than Elo:
  - Models the number of goals, not just win/draw/loss.
  - Captures separate offensive and defensive abilities.
  - Time-decay weighting means recent form influences the model more.
  - Dixon-Coles rho correction fixes the systematic under-prediction of draws
    in low-scoring matches (0-0, 0-1, 1-0, 1-1).
  - Form multiplier adjusts expected goals based on a team's last 6 results
    in the current season.
  - Poisson score sampling is more realistic than a three-outcome coin-flip.
"""

import json
import math
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIME_DECAY   = 0.003   # e-fold decay rate in days  (half-weight ≈ 231 days)
FORM_GAMES   = 6       # games used to compute current-season form
FORM_WEIGHT  = 0.20    # how strongly form shifts expected goals  (±10 % per unit)
N_GOAL_CAP   = 10      # max goals per team in simulation (sanity cap)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_season_data(path: str = "data/matches.json"):
    """
    Return (completed, upcoming) DataFrames.
    Completed rows have home_goals / away_goals.
    Upcoming rows have no score.
    """
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
                done.append({**base,
                             "home_goals": int(ft[0]),
                             "away_goals": int(ft[1])})
            except (ValueError, TypeError):
                pass
        else:
            todo.append(base)

    _ecols = ["season", "date", "home_team", "away_team", "home_goals", "away_goals"]
    completed = pd.DataFrame(done) if done else pd.DataFrame(columns=_ecols)
    upcoming  = pd.DataFrame(todo) if todo else pd.DataFrame(
        columns=["season", "date", "home_team", "away_team"])

    if not completed.empty:
        completed["date"] = pd.to_datetime(completed["date"], errors="coerce")
        completed = completed.sort_values("date").reset_index(drop=True)

    return completed, upcoming


# ---------------------------------------------------------------------------
# Dixon-Coles Poisson model fitting
# ---------------------------------------------------------------------------

def fit_dixoncoles(completed: pd.DataFrame,
                   time_decay: float = TIME_DECAY) -> tuple:
    """
    Fit a time-weighted Dixon-Coles Poisson model to all completed matches.

    Each team gets two parameters:
      attack[i]  — log attacking strength  (0 = league-average)
      defense[i] — log defensive strength  (higher = harder to score against)

    Plus global parameters:
      home_adv — log home-field advantage
      rho      — Dixon-Coles low-score correlation correction (typically ≈ −0.1)

    Returns
    -------
    attack  : {team: float}
    defense : {team: float}
    home_adv: float
    rho     : float
    teams   : list[str]
    """
    teams = sorted(set(
        completed["home_team"].tolist() + completed["away_team"].tolist()
    ))
    n   = len(teams)
    idx = {t: i for i, t in enumerate(teams)}

    ref_date  = completed["date"].max()
    days_ago  = (ref_date - completed["date"]).dt.days.fillna(730).clip(0, 1460).values
    weights   = np.exp(-time_decay * days_ago)

    hi_arr = np.array([idx[t] for t in completed["home_team"]])
    ai_arr = np.array([idx[t] for t in completed["away_team"]])
    hg_arr = completed["home_goals"].values.astype(np.int32)
    ag_arr = completed["away_goals"].values.astype(np.int32)

    def neg_ll(params: np.ndarray) -> float:
        atk = params[:n]
        dfd = params[n:2 * n]
        hadv = params[2 * n]
        rho  = float(np.clip(params[2 * n + 1], -0.4, 0.0))

        lam = np.exp(hadv + atk[hi_arr] - dfd[ai_arr])   # expected home goals
        mu  = np.exp(atk[ai_arr] - dfd[hi_arr])           # expected away goals

        # Clamp to avoid log(0)
        lam = np.maximum(lam, 1e-9)
        mu  = np.maximum(mu,  1e-9)

        # Dixon-Coles tau correction (vectorised)
        tau = np.ones(len(lam))
        m00 = (hg_arr == 0) & (ag_arr == 0)
        m01 = (hg_arr == 0) & (ag_arr == 1)
        m10 = (hg_arr == 1) & (ag_arr == 0)
        m11 = (hg_arr == 1) & (ag_arr == 1)
        tau[m00] = np.maximum(1.0 - lam[m00] * mu[m00] * rho, 1e-9)
        tau[m01] = 1.0 + lam[m01] * rho
        tau[m10] = 1.0 + mu[m10]  * rho
        tau[m11] = 1.0 - rho

        # Poisson log-PMF (vectorised via scipy.special.gammaln)
        ll_home = hg_arr * np.log(lam) - lam - gammaln(hg_arr + 1)
        ll_away = ag_arr * np.log(mu)  - mu  - gammaln(ag_arr + 1)

        ll = weights * (np.log(np.maximum(tau, 1e-9)) + ll_home + ll_away)

        # L2 regularisation: keeps parameters identifiable and stable
        reg = 5e-4 * np.sum(params[:2 * n] ** 2)

        return -(ll.sum()) + reg

    # Initial guess
    x0 = np.zeros(2 * n + 2)
    x0[2 * n]     =  0.25   # home advantage  ~ exp(0.25) ≈ 1.28
    x0[2 * n + 1] = -0.08   # rho

    res = minimize(neg_ll, x0, method="L-BFGS-B",
                   options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8})

    p = res.x
    attack  = {teams[i]: float(p[i])     for i in range(n)}
    defense = {teams[i]: float(p[n + i]) for i in range(n)}
    home_adv = float(p[2 * n])
    rho      = float(np.clip(p[2 * n + 1], -0.4, 0.0))

    return attack, defense, home_adv, rho, teams


# ---------------------------------------------------------------------------
# Elo history (for visualisation)
# ---------------------------------------------------------------------------

def compute_elo_history(completed: pd.DataFrame,
                        k: int = 32, home_advantage: float = 50.0) -> pd.DataFrame:
    """Return per-match Elo rating history for all teams."""
    START = 1500.0
    ratings: dict[str, float] = defaultdict(lambda: START)
    rows = []

    for _, row in completed.iterrows():
        h, a = row["home_team"], row["away_team"]
        rh, ra = ratings[h], ratings[a]

        if row["home_goals"] > row["away_goals"]:
            sh, sa = 1.0, 0.0
        elif row["home_goals"] < row["away_goals"]:
            sh, sa = 0.0, 1.0
        else:
            sh = sa = 0.5

        eh = 1.0 / (1.0 + 10.0 ** ((ra - rh - home_advantage) / 400.0))
        ea = 1.0 - eh

        ratings[h] = rh + k * (sh - eh)
        ratings[a] = ra + k * (sa - ea)

        rows.append({"date": row["date"], "season": row["season"],
                     "team": h, "rating": ratings[h]})
        rows.append({"date": row["date"], "season": row["season"],
                     "team": a, "rating": ratings[a]})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Current-season standings
# ---------------------------------------------------------------------------

def compute_standings(completed: pd.DataFrame,
                      season: str) -> tuple[dict, dict, dict]:
    """Return (points, goal_diff, goals_for) for all teams in season."""
    pts: dict[str, int] = defaultdict(int)
    gd:  dict[str, int] = defaultdict(int)
    gf:  dict[str, int] = defaultdict(int)

    df = completed[completed["season"] == season]
    for _, row in df.iterrows():
        hg, ag = int(row["home_goals"]), int(row["away_goals"])
        h,  a  = row["home_team"],       row["away_team"]

        if hg > ag:
            pts[h] += 3
        elif hg < ag:
            pts[a] += 3
        else:
            pts[h] += 1; pts[a] += 1

        gd[h] += hg - ag;  gd[a] += ag - hg
        gf[h] += hg;       gf[a] += ag

    return dict(pts), dict(gd), dict(gf)


# ---------------------------------------------------------------------------
# Form calculation
# ---------------------------------------------------------------------------

def compute_form(completed: pd.DataFrame,
                 season: str,
                 n_games: int = FORM_GAMES) -> tuple[dict, dict]:
    """
    Compute form score (0-1) and result string ('WLDWW') for each team
    based on their last n_games in the target season.

    A form score of 1.0 = perfect (all wins), 0.5 = half points, 0.0 = zero points.
    """
    df = completed[completed["season"] == season].sort_values("date")
    all_teams = set(df["home_team"].tolist()) | set(df["away_team"].tolist())

    form_scores:  dict[str, float] = {}
    form_strings: dict[str, str]   = {}

    for team in all_teams:
        mask   = (df["home_team"] == team) | (df["away_team"] == team)
        recent = df[mask].tail(n_games)

        results = []
        for _, row in recent.iterrows():
            is_home = (row["home_team"] == team)
            hg, ag  = int(row["home_goals"]), int(row["away_goals"])
            if is_home:
                res = "W" if hg > ag else ("L" if hg < ag else "D")
            else:
                res = "W" if ag > hg else ("L" if ag < hg else "D")
            results.append(res)

        if results:
            pts_earned = sum(3 if r == "W" else 1 if r == "D" else 0 for r in results)
            max_pts    = len(results) * 3
            form_scores[team]  = pts_earned / max_pts
            form_strings[team] = "".join(results)
        else:
            form_scores[team]  = 0.5
            form_strings[team] = ""

    return form_scores, form_strings


# ---------------------------------------------------------------------------
# Match simulation (Dixon-Coles Poisson)
# ---------------------------------------------------------------------------

def _dc_tau(hg: int, ag: int, lam: float, mu: float, rho: float) -> float:
    """Dixon-Coles correction factor for low-scoring outcomes."""
    if   hg == 0 and ag == 0: return max(1.0 - lam * mu * rho, 1e-9)
    elif hg == 0 and ag == 1: return 1.0 + lam * rho
    elif hg == 1 and ag == 0: return 1.0 + mu  * rho
    elif hg == 1 and ag == 1: return 1.0 - rho
    return 1.0


def simulate_match(
    attack_h: float, defense_h: float,
    attack_a: float, defense_a: float,
    home_adv: float, rho: float,
    form_h: float, form_a: float,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """
    Sample a single match score using time-weighted Dixon-Coles Poisson model.

    form_h / form_a: 0-1 form score; 0.5 = neutral.
    """
    # Expected goals from the fitted model
    lam = math.exp(home_adv + attack_h - defense_a)
    mu  = math.exp(attack_a - defense_h)

    # Apply form multiplier  (±FORM_WEIGHT when on perfect / zero form)
    lam *= max(0.4, 1.0 + FORM_WEIGHT * (form_h - 0.5) * 2.0)
    mu  *= max(0.4, 1.0 + FORM_WEIGHT * (form_a - 0.5) * 2.0)

    lam = max(0.05, lam)
    mu  = max(0.05, mu)

    # Dixon-Coles rejection sampling for low-scoring outcomes
    while True:
        hg = min(int(rng.poisson(lam)), N_GOAL_CAP)
        ag = min(int(rng.poisson(mu)),  N_GOAL_CAP)
        tau = _dc_tau(hg, ag, lam, mu, rho)
        # Accept with probability tau (normalised; max tau is ~1 so rarely reject)
        if rng.random() < tau:
            return hg, ag


# ---------------------------------------------------------------------------
# Monte Carlo simulation
# ---------------------------------------------------------------------------

def run_monte_carlo(
    completed: pd.DataFrame,
    upcoming:  pd.DataFrame,
    target_season: str,
    n_simulations: int = 10_000,
) -> tuple[dict, dict, dict, dict, list]:
    """
    Simulate remaining fixtures in target_season using the Dixon-Coles model.

    Returns
    -------
    win_probs   : {team: float}  — fraction of simulations where team finishes 1st
    cur_points  : {team: int}    — actual points so far
    cur_gd      : {team: int}    — actual goal difference so far
    elo_ratings : {team: float}  — simple Elo ratings (for the Elo chart)
    teams       : list[str]
    """
    print("  Fitting Dixon-Coles Poisson model...", end=" ", flush=True)
    attack, defense, home_adv, rho, all_teams = fit_dixoncoles(completed)
    print("done")

    cur_points, cur_gd, _ = compute_standings(completed, target_season)
    form_scores, _         = compute_form(completed, target_season)

    remaining = upcoming[upcoming["season"] == target_season].copy()

    # Collect all teams present in this season
    s_done = completed[completed["season"] == target_season]
    s_todo = remaining
    season_teams: set[str] = set()
    for df in (s_done, s_todo):
        for col in ("home_team", "away_team"):
            if col in df.columns:
                season_teams.update(df[col].dropna().tolist())

    for t in season_teams:
        cur_points.setdefault(t, 0)

    DEFAULT_ATK = float(np.mean(list(attack.values()))) if attack else 0.0
    DEFAULT_DEF = float(np.mean(list(defense.values()))) if defense else 0.0

    # --- Season already complete ---
    if remaining.empty:
        winner = max(season_teams, key=lambda t: (cur_points.get(t, 0), cur_gd.get(t, 0)))
        probs  = {t: 0.0 for t in season_teams}
        probs[winner] = 1.0
        elo_ratings = _simple_elo(completed)
        return probs, cur_points, cur_gd, elo_ratings, sorted(season_teams)

    remaining_records = remaining.to_dict("records")
    rng  = np.random.default_rng(42)
    wins: dict[str, int] = defaultdict(int)

    for _ in range(n_simulations):
        sim_pts = dict(cur_points)

        for match in remaining_records:
            h, a = match["home_team"], match["away_team"]
            atk_h = attack.get(h,  DEFAULT_ATK)
            def_h = defense.get(h, DEFAULT_DEF)
            atk_a = attack.get(a,  DEFAULT_ATK)
            def_a = defense.get(a, DEFAULT_DEF)
            fh    = form_scores.get(h, 0.5)
            fa    = form_scores.get(a, 0.5)

            hg, ag = simulate_match(atk_h, def_h, atk_a, def_a,
                                    home_adv, rho, fh, fa, rng)
            if hg > ag:
                sim_pts[h] = sim_pts.get(h, 0) + 3
            elif hg < ag:
                sim_pts[a] = sim_pts.get(a, 0) + 3
            else:
                sim_pts[h] = sim_pts.get(h, 0) + 1
                sim_pts[a] = sim_pts.get(a, 0) + 1

        winner = max(season_teams, key=lambda t: sim_pts.get(t, 0))
        wins[winner] += 1

    win_probs = {t: wins.get(t, 0) / n_simulations for t in season_teams}
    elo_ratings = _simple_elo(completed)

    return win_probs, cur_points, cur_gd, elo_ratings, sorted(season_teams)


def _simple_elo(completed: pd.DataFrame, k: int = 32) -> dict:
    """Lightweight Elo for display purposes (passed to the Elo chart)."""
    START = 1500.0
    ratings: dict[str, float] = defaultdict(lambda: START)
    for _, row in completed.iterrows():
        h, a = row["home_team"], row["away_team"]
        rh, ra = ratings[h], ratings[a]
        sh = 1.0 if row["home_goals"] > row["away_goals"] else (0.0 if row["home_goals"] < row["away_goals"] else 0.5)
        sa = 1.0 - sh
        eh = 1.0 / (1.0 + 10.0 ** ((ra - rh - 50) / 400.0))
        ratings[h] = rh + k * (sh - eh)
        ratings[a] = ra + k * (sa - (1.0 - eh))
    return dict(ratings)
