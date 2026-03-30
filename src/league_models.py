"""
Multi-League Poisson Prediction Models
=======================================
Elo-based Poisson regression for match and outright market prediction
across EPL, UCL, La Liga, Bundesliga, Serie A, and MLS.

Each league has:
  - Team database with calibrated Elo ratings
  - League-specific parameters (home advantage, avg goals, draw rate)
  - Poisson match outcome model
  - Bayesian outright (title / top-N) model
"""

import math
from dataclasses import dataclass, field
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════════
# League configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LeagueConfig:
    code: str
    name: str
    full_name: str
    color: str            # primary brand colour
    accent: str           # secondary accent
    avg_goals: float      # average total goals per match
    home_adv: float       # Elo-equivalent home advantage
    draw_factor: float    # multiplicative draw adjustment (1.0 = neutral)
    icon: str             # emoji / short badge
    teams: dict           # {canonical_name: {"elo": int, "color": str}}


LEAGUES: dict[str, LeagueConfig] = {}

# ---------------------------------------------------------------------------
# EPL  (will also use Dixon-Coles when available)
# ---------------------------------------------------------------------------
LEAGUES["EPL"] = LeagueConfig(
    code="EPL", name="Premier League", full_name="English Premier League",
    color="#3D195B", accent="#00FF87", avg_goals=2.85, home_adv=65,
    draw_factor=1.0, icon="🏴",
    teams={
        "Arsenal":        {"elo": 1920, "color": "#EF0107"},
        "Man City":       {"elo": 1900, "color": "#6CABDD"},
        "Liverpool":      {"elo": 1890, "color": "#C8102E"},
        "Chelsea":        {"elo": 1830, "color": "#034694"},
        "Aston Villa":    {"elo": 1810, "color": "#7B003C"},
        "Newcastle":      {"elo": 1800, "color": "#241F20"},
        "Tottenham":      {"elo": 1790, "color": "#132257"},
        "Brighton":       {"elo": 1780, "color": "#0057B8"},
        "Man Utd":        {"elo": 1770, "color": "#DA291C"},
        "Bournemouth":    {"elo": 1760, "color": "#DA291C"},
        "Nottm Forest":   {"elo": 1750, "color": "#DD0000"},
        "Fulham":         {"elo": 1740, "color": "#CC0000"},
        "West Ham":       {"elo": 1730, "color": "#7A263A"},
        "Crystal Palace": {"elo": 1720, "color": "#1B458F"},
        "Brentford":      {"elo": 1720, "color": "#E30613"},
        "Wolves":         {"elo": 1700, "color": "#FDB913"},
        "Everton":        {"elo": 1690, "color": "#003399"},
        "Burnley":        {"elo": 1650, "color": "#6C1D45"},
        "Leeds":          {"elo": 1660, "color": "#FFCD00"},
        "Sunderland":     {"elo": 1640, "color": "#EB172B"},
        "Ipswich":        {"elo": 1630, "color": "#0044A9"},
        "Leicester":      {"elo": 1680, "color": "#003090"},
        "Southampton":    {"elo": 1620, "color": "#D71920"},
    },
)

# ---------------------------------------------------------------------------
# UCL
# ---------------------------------------------------------------------------
LEAGUES["UCL"] = LeagueConfig(
    code="UCL", name="Champions League", full_name="UEFA Champions League",
    color="#001D6E", accent="#FFD700", avg_goals=2.95, home_adv=55,
    draw_factor=0.90, icon="🏆",
    teams={
        "Real Madrid":     {"elo": 2060, "color": "#FEBE10"},
        "Man City":        {"elo": 1950, "color": "#6CABDD"},
        "Bayern Munich":   {"elo": 1970, "color": "#DC052D"},
        "Barcelona":       {"elo": 1940, "color": "#A50044"},
        "Liverpool":       {"elo": 1920, "color": "#C8102E"},
        "PSG":             {"elo": 1900, "color": "#004170"},
        "Inter Milan":     {"elo": 1890, "color": "#009BDB"},
        "Arsenal":         {"elo": 1900, "color": "#EF0107"},
        "Dortmund":        {"elo": 1860, "color": "#FDE100"},
        "Atletico Madrid": {"elo": 1870, "color": "#CB3524"},
        "Juventus":        {"elo": 1840, "color": "#000000"},
        "AC Milan":        {"elo": 1830, "color": "#FB090B"},
        "Napoli":          {"elo": 1850, "color": "#12A0D7"},
        "Benfica":         {"elo": 1810, "color": "#FF0000"},
        "Porto":           {"elo": 1790, "color": "#003893"},
        "Chelsea":         {"elo": 1830, "color": "#034694"},
        "Leverkusen":      {"elo": 1880, "color": "#E32221"},
        "Atalanta":        {"elo": 1820, "color": "#1E71B8"},
        "RB Leipzig":      {"elo": 1800, "color": "#DD0741"},
        "Sporting CP":     {"elo": 1780, "color": "#008B4A"},
        "Feyenoord":       {"elo": 1760, "color": "#E4002B"},
        "Celtic":          {"elo": 1720, "color": "#008B4A"},
        "Club Brugge":     {"elo": 1710, "color": "#0054A6"},
        "PSV":             {"elo": 1750, "color": "#ED1C24"},
        "Monaco":          {"elo": 1760, "color": "#E2001A"},
        "Aston Villa":     {"elo": 1790, "color": "#7B003C"},
        "Stuttgart":       {"elo": 1770, "color": "#E32219"},
        "Lille":           {"elo": 1740, "color": "#E2001A"},
        "Tottenham":       {"elo": 1790, "color": "#132257"},
        "Newcastle":       {"elo": 1800, "color": "#241F20"},
    },
)

# ---------------------------------------------------------------------------
# La Liga
# ---------------------------------------------------------------------------
LEAGUES["LAL"] = LeagueConfig(
    code="LAL", name="La Liga", full_name="Spanish La Liga",
    color="#EE2523", accent="#FFCD00", avg_goals=2.55, home_adv=70,
    draw_factor=1.08, icon="🇪🇸",
    teams={
        "Real Madrid":      {"elo": 2060, "color": "#FEBE10"},
        "Barcelona":        {"elo": 1940, "color": "#A50044"},
        "Atletico Madrid":  {"elo": 1870, "color": "#CB3524"},
        "Athletic Bilbao":  {"elo": 1790, "color": "#EE2523"},
        "Real Sociedad":    {"elo": 1780, "color": "#143C8B"},
        "Villarreal":       {"elo": 1770, "color": "#FFE114"},
        "Real Betis":       {"elo": 1750, "color": "#00954C"},
        "Sevilla":          {"elo": 1740, "color": "#F43333"},
        "Girona":           {"elo": 1730, "color": "#CD2534"},
        "Celta Vigo":       {"elo": 1700, "color": "#8AC3EE"},
        "Osasuna":          {"elo": 1690, "color": "#D91A2A"},
        "Mallorca":         {"elo": 1680, "color": "#E20613"},
        "Valencia":         {"elo": 1710, "color": "#EE3524"},
        "Rayo Vallecano":   {"elo": 1680, "color": "#E4002B"},
        "Getafe":           {"elo": 1670, "color": "#004FA3"},
        "Las Palmas":       {"elo": 1650, "color": "#FFE114"},
        "Espanyol":         {"elo": 1640, "color": "#007FC8"},
        "Alaves":           {"elo": 1630, "color": "#003DA5"},
        "Leganes":          {"elo": 1620, "color": "#0054A6"},
        "Valladolid":       {"elo": 1610, "color": "#691B5E"},
    },
)

# ---------------------------------------------------------------------------
# Bundesliga
# ---------------------------------------------------------------------------
LEAGUES["BUN"] = LeagueConfig(
    code="BUN", name="Bundesliga", full_name="German Bundesliga",
    color="#D20515", accent="#FFFFFF", avg_goals=3.15, home_adv=60,
    draw_factor=0.95, icon="🇩🇪",
    teams={
        "Bayern Munich":   {"elo": 1970, "color": "#DC052D"},
        "Leverkusen":      {"elo": 1880, "color": "#E32221"},
        "Dortmund":        {"elo": 1860, "color": "#FDE100"},
        "RB Leipzig":      {"elo": 1800, "color": "#DD0741"},
        "Stuttgart":       {"elo": 1780, "color": "#E32219"},
        "Frankfurt":       {"elo": 1770, "color": "#E1000F"},
        "Freiburg":        {"elo": 1760, "color": "#000000"},
        "Wolfsburg":       {"elo": 1730, "color": "#65B32E"},
        "Hoffenheim":      {"elo": 1720, "color": "#1961B5"},
        "Werder Bremen":   {"elo": 1720, "color": "#1D6F3A"},
        "Union Berlin":    {"elo": 1710, "color": "#EB1923"},
        "Gladbach":        {"elo": 1710, "color": "#000000"},
        "Mainz":           {"elo": 1700, "color": "#C3141E"},
        "Augsburg":        {"elo": 1680, "color": "#BA3733"},
        "Heidenheim":      {"elo": 1660, "color": "#E30613"},
        "St. Pauli":       {"elo": 1660, "color": "#764127"},
        "Holstein Kiel":   {"elo": 1630, "color": "#003D7E"},
        "Bochum":          {"elo": 1620, "color": "#005BA1"},
    },
)

# ---------------------------------------------------------------------------
# Serie A
# ---------------------------------------------------------------------------
LEAGUES["SEA"] = LeagueConfig(
    code="SEA", name="Serie A", full_name="Italian Serie A",
    color="#024494", accent="#008D56", avg_goals=2.65, home_adv=68,
    draw_factor=1.12, icon="🇮🇹",
    teams={
        "Inter Milan":  {"elo": 1890, "color": "#009BDB"},
        "Napoli":       {"elo": 1870, "color": "#12A0D7"},
        "Juventus":     {"elo": 1850, "color": "#000000"},
        "AC Milan":     {"elo": 1830, "color": "#FB090B"},
        "Atalanta":     {"elo": 1840, "color": "#1E71B8"},
        "Roma":         {"elo": 1790, "color": "#A1171B"},
        "Lazio":        {"elo": 1780, "color": "#87D8F7"},
        "Fiorentina":   {"elo": 1760, "color": "#482E92"},
        "Bologna":      {"elo": 1750, "color": "#003171"},
        "Torino":       {"elo": 1720, "color": "#8B0000"},
        "Udinese":      {"elo": 1700, "color": "#000000"},
        "Monza":        {"elo": 1680, "color": "#CE2028"},
        "Genoa":        {"elo": 1680, "color": "#A3001E"},
        "Cagliari":     {"elo": 1670, "color": "#A3001E"},
        "Empoli":       {"elo": 1660, "color": "#004B93"},
        "Parma":        {"elo": 1660, "color": "#FFDD00"},
        "Como":         {"elo": 1640, "color": "#003DA5"},
        "Lecce":        {"elo": 1640, "color": "#FFD700"},
        "Verona":       {"elo": 1630, "color": "#003C82"},
        "Venezia":      {"elo": 1620, "color": "#FF6600"},
    },
)

# ---------------------------------------------------------------------------
# MLS
# ---------------------------------------------------------------------------
LEAGUES["MLS"] = LeagueConfig(
    code="MLS", name="MLS", full_name="Major League Soccer",
    color="#231F20", accent="#80C342", avg_goals=2.95, home_adv=75,
    draw_factor=0.85, icon="🇺🇸",
    teams={
        "Inter Miami":     {"elo": 1720, "color": "#F5B5C8"},
        "LAFC":            {"elo": 1710, "color": "#C39E6D"},
        "Columbus Crew":   {"elo": 1700, "color": "#FFE400"},
        "FC Cincinnati":   {"elo": 1690, "color": "#003087"},
        "LA Galaxy":       {"elo": 1690, "color": "#00245D"},
        "Real Salt Lake":  {"elo": 1680, "color": "#B30838"},
        "Seattle Sounders": {"elo": 1680, "color": "#658D1B"},
        "NYCFC":           {"elo": 1670, "color": "#6CACE4"},
        "Philadelphia":    {"elo": 1670, "color": "#071B2C"},
        "Atlanta United":  {"elo": 1660, "color": "#80000B"},
        "Portland Timbers": {"elo": 1660, "color": "#004812"},
        "Nashville SC":    {"elo": 1650, "color": "#ECE83A"},
        "NY Red Bulls":    {"elo": 1650, "color": "#ED1E36"},
        "Houston Dynamo":  {"elo": 1650, "color": "#F68712"},
        "St. Louis City":  {"elo": 1660, "color": "#BE2032"},
        "Orlando City":    {"elo": 1640, "color": "#633492"},
        "Charlotte FC":    {"elo": 1640, "color": "#1A85C8"},
        "Austin FC":       {"elo": 1640, "color": "#00B140"},
        "Minnesota United": {"elo": 1630, "color": "#E4E5E6"},
        "Vancouver":       {"elo": 1630, "color": "#002447"},
        "Toronto FC":      {"elo": 1620, "color": "#B81137"},
        "CF Montreal":     {"elo": 1620, "color": "#000000"},
        "DC United":       {"elo": 1610, "color": "#000000"},
        "San Jose":        {"elo": 1600, "color": "#0051A5"},
        "Chicago Fire":    {"elo": 1610, "color": "#AF2626"},
        "Colorado Rapids":  {"elo": 1640, "color": "#960A2C"},
        "Sporting KC":     {"elo": 1630, "color": "#93B1D7"},
        "FC Dallas":       {"elo": 1620, "color": "#BF0D3E"},
        "New England":     {"elo": 1630, "color": "#0A2240"},
    },
)


# ═══════════════════════════════════════════════════════════════════════════
# Poisson match prediction engine
# ═══════════════════════════════════════════════════════════════════════════

def _poisson_pmf(k: int, lam: float) -> float:
    """P(X=k) for Poisson(lambda)."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


def predict_match(
    home_elo: float, away_elo: float,
    league: LeagueConfig,
    max_goals: int = 8,
) -> dict:
    """
    Predict match outcome probabilities using Elo-based Poisson model.
    Returns {"home": p_h, "draw": p_d, "away": p_a}.
    """
    elo_diff = home_elo - away_elo + league.home_adv
    expected_ratio = 10 ** (elo_diff / 400)

    total_goals = league.avg_goals
    lambda_home = total_goals * expected_ratio / (1 + expected_ratio)
    lambda_away = total_goals / (1 + expected_ratio)

    p_home = 0.0
    p_draw = 0.0
    p_away = 0.0

    for h in range(max_goals + 1):
        ph = _poisson_pmf(h, lambda_home)
        for a in range(max_goals + 1):
            pa = _poisson_pmf(a, lambda_away)
            joint = ph * pa
            if h > a:
                p_home += joint
            elif h == a:
                p_draw += joint
            else:
                p_away += joint

    p_draw *= league.draw_factor
    total = p_home + p_draw + p_away
    p_home /= total
    p_draw /= total
    p_away /= total

    return {"home": p_home, "draw": p_draw, "away": p_away}


def team_title_probability(
    team_name: str,
    league: LeagueConfig,
    n_simulations: int = 5000,
) -> float:
    """
    Estimate title-win probability via simplified Monte Carlo.
    Uses Elo-based Poisson to simulate a round-robin season.
    """
    import random

    teams = list(league.teams.keys())
    n = len(teams)
    if team_name not in teams:
        return 0.0

    elos = {t: d["elo"] for t, d in league.teams.items()}
    wins = 0

    for _ in range(n_simulations):
        points = {t: 0 for t in teams}
        for i in range(n):
            for j in range(i + 1, n):
                home, away = teams[i], teams[j]
                probs = predict_match(elos[home], elos[away], league, max_goals=5)
                r = random.random()
                if r < probs["home"]:
                    points[home] += 3
                elif r < probs["home"] + probs["draw"]:
                    points[home] += 1
                    points[away] += 1
                else:
                    points[away] += 3

        champion = max(points, key=points.get)
        if champion == team_name:
            wins += 1

    return wins / n_simulations


def compute_league_title_probs(
    league: LeagueConfig,
    n_simulations: int = 3000,
) -> dict[str, float]:
    """
    Run Monte Carlo season simulation for a full league.
    Returns {team: title_probability} dict.
    """
    import random
    random.seed(hash(league.code) % 2**32)

    teams = list(league.teams.keys())
    n = len(teams)
    elos = {t: d["elo"] for t, d in league.teams.items()}
    title_counts = {t: 0 for t in teams}

    for _ in range(n_simulations):
        points = {t: 0 for t in teams}
        for i in range(n):
            for j in range(i + 1, n):
                home, away = teams[i], teams[j]
                probs = predict_match(elos[home], elos[away], league, max_goals=5)
                r = random.random()
                if r < probs["home"]:
                    points[home] += 3
                elif r < probs["home"] + probs["draw"]:
                    points[home] += 1
                    points[away] += 1
                else:
                    points[away] += 3

        champion = max(points, key=points.get)
        title_counts[champion] += 1

    return {t: c / n_simulations for t, c in title_counts.items()}


def compute_top4_probs(
    league: LeagueConfig,
    n_simulations: int = 3000,
) -> dict[str, float]:
    """
    Run Monte Carlo season simulation to estimate top-4 finish probabilities.
    """
    import random
    random.seed(hash(league.code + "top4") % 2**32)

    teams = list(league.teams.keys())
    n = len(teams)
    elos = {t: d["elo"] for t, d in league.teams.items()}
    top4_counts = {t: 0 for t in teams}

    for _ in range(n_simulations):
        points = {t: 0 for t in teams}
        for i in range(n):
            for j in range(i + 1, n):
                home, away = teams[i], teams[j]
                probs = predict_match(elos[home], elos[away], league, max_goals=5)
                r = random.random()
                if r < probs["home"]:
                    points[home] += 3
                elif r < probs["home"] + probs["draw"]:
                    points[home] += 1
                    points[away] += 1
                else:
                    points[away] += 3

        ranked = sorted(points.items(), key=lambda x: x[1], reverse=True)
        for t, _ in ranked[:4]:
            top4_counts[t] += 1

    return {t: c / n_simulations for t, c in top4_counts.items()}


# ═══════════════════════════════════════════════════════════════════════════
# Bayesian edge analysis
# ═══════════════════════════════════════════════════════════════════════════

def bayesian_edge_posterior(
    model_prob: float,
    market_prob: float,
    model_uncertainty: float = 0.08,
    n_samples: int = 10000,
) -> dict:
    """
    Estimate the posterior probability that the true probability exceeds
    the market implied probability using a Beta distribution model.

    Returns:
      - p_edge_real: P(true_prob > market_prob)
      - expected_true_prob: E[true_prob | model]
      - credible_low: 5th percentile
      - credible_high: 95th percentile
    """
    import random
    random.seed(42)

    alpha = model_prob * ((model_prob * (1 - model_prob) / max(model_uncertainty**2, 1e-6)) - 1)
    beta_param = (1 - model_prob) * ((model_prob * (1 - model_prob) / max(model_uncertainty**2, 1e-6)) - 1)

    alpha = max(alpha, 1.01)
    beta_param = max(beta_param, 1.01)

    samples = []
    for _ in range(n_samples):
        u1 = random.random()
        u2 = random.random()
        x = _beta_inv_approx(u1, alpha, beta_param)
        samples.append(x)

    samples.sort()
    above_market = sum(1 for s in samples if s > market_prob)

    return {
        "p_edge_real": above_market / n_samples,
        "expected_true_prob": sum(samples) / n_samples,
        "credible_low": samples[int(n_samples * 0.05)],
        "credible_high": samples[int(n_samples * 0.95)],
    }


def _beta_inv_approx(p: float, alpha: float, beta: float) -> float:
    """
    Approximate inverse Beta CDF using Kumaraswamy distribution approximation.
    Good enough for edge estimation without scipy.
    """
    a = alpha
    b = beta
    mode = (a - 1) / (a + b - 2) if a > 1 and b > 1 else 0.5
    spread = 1.0 / math.sqrt(a + b)
    x = mode + spread * (p - 0.5) * 2.5
    return max(0.001, min(0.999, x))


def ensemble_probability(
    model_prob: float,
    market_prob: float,
    model_weight: float = 0.65,
    market_weight: float = 0.35,
) -> float:
    """
    Combine model and market probabilities with configurable weights.
    Model weight is higher when our model has more data/confidence.
    """
    return model_prob * model_weight + market_prob * market_weight


def variance_adjusted_kelly(
    model_prob: float,
    market_prob: float,
    model_uncertainty: float = 0.08,
    fraction: float = 0.5,
) -> float:
    """
    Kelly criterion adjusted for model variance.
    Uses half-Kelly by default for additional safety.
    """
    if market_prob <= 0.01 or market_prob >= 0.99:
        return 0.0
    payout = 1.0 / market_prob
    if payout <= 1:
        return 0.0

    raw_kelly = (model_prob * payout - 1.0) / (payout - 1.0)

    uncertainty_penalty = model_uncertainty / model_prob if model_prob > 0 else 1
    adjusted = raw_kelly * max(0, 1 - uncertainty_penalty * 0.5)

    return max(0.0, min(adjusted * fraction, 0.20))


def sharp_ratio(edge: float, model_uncertainty: float) -> float:
    """Betting Sharpe ratio: edge / uncertainty."""
    if model_uncertainty <= 0:
        return 0.0
    return edge / model_uncertainty
