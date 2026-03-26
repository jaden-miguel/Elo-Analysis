# Premier League Title-Race Prediction (2022-2026)

This project fetches live Premier League match data across multiple seasons,
fits a statistically rigorous prediction model, and produces an interactive
HTML dashboard that forecasts which team will win the current season's title.

---

## Features

- Live match data from [openfootball](https://github.com/openfootball/football.json) (2022-23 through the current season)
- Dixon-Coles time-weighted Poisson model — the same family of models used by professional football analytics teams
- **Model v3:** separate **home and away** attack/defence parameters per club (four ratings each)
- Time decay plus **2× weight** on the current season so live form dominates
- Goals-based form (last 8 matches, exponentially weighted) adjusts expected goals
- **Head-to-head** adjustment from the last six direct meetings (damped)
- **50,000** vectorised Monte Carlo simulations (title, top 4, relegation)
- Low-score Dixon-Coles **rho** correction on analytical match markets
- Elo rating tracker across all seasons for trend visualisation
- HTML dashboard: title race, **season markets** (fair odds), **best bets** ranking, upcoming fixture breakdown (1X2, Over 2.5, BTTS), standings with form badges
- No paid API key required

---

## Project Structure

```
Elo-Analysis-main/
├── data/
│   ├── matches.json                  Raw match results (all seasons)
│   └── prediction_dashboard.html     Generated prediction dashboard
├── src/
│   ├── data_fetch.py                 Downloads and merges match data
│   ├── data_processing.py            Aggregates team points from raw data
│   ├── elo_rating.py                 Basic Elo rating module
│   ├── analysis.py                   Generates summary CSVs
│   ├── prediction.py                 Dixon-Coles Poisson model + Monte Carlo engine
│   ├── predict_dashboard.py          Builds the HTML prediction dashboard
│   └── visualization.py             Legacy Plotly charts (points + Elo)
├── assets/
│   └── example_output.png            Chart screenshot
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Clone and install dependencies

```bash
git clone <this-repo>.git
cd Elo-Analysis-main
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Fetch match data

```bash
python src/data_fetch.py
```

This downloads all available seasons from openfootball (2022-23 through
2025-26) and saves them to `data/matches.json`. Seasons not yet available
are skipped automatically.

### 3. Run the prediction dashboard

```bash
python src/predict_dashboard.py
```

This will:
1. Fit the Dixon-Coles v3 model (home/away split) to all completed matches
2. Run 50,000 Monte Carlo simulations of remaining fixtures
3. Generate `data/prediction_dashboard.html`
4. Open the dashboard automatically in your default browser

---

## The Prediction Model

### Why Dixon-Coles over Elo?

Standard Elo tracks only win/draw/loss. The Dixon-Coles model instead models
the number of goals each team scores, which contains significantly more
information.

Each team has **four** fitted log-strength parameters:

| Parameter | Meaning |
|-----------|---------|
| `atk_home[team]` | Attacking strength when playing at home |
| `atk_away[team]` | Attacking strength when playing away |
| `def_home[team]` | Defensive quality when hosting |
| `def_away[team]` | Defensive quality when travelling |

Expected goals for a fixture:

- Home: `exp(home_adv + atk_home[home] - def_away[away])`
- Away: `exp(atk_away[away] - def_home[home])`

Global parameters: `home_adv` and Dixon-Coles `rho` (low-score correction on
analytical markets; simulations use fast vectorised Poisson draws).

### Time weighting

Each match is weighted by `exp(-0.002 * days_ago)` times a **2× multiplier**
if the row belongs to the most recent season in the dataset. Recent and
current-season data therefore dominate older seasons.

### Form adjustment

Over the last **8** games in the current season, goals scored and conceded are
averaged with exponential intra-window weighting (newest games count most).
These ratios feed multipliers on attack and defence (damped with `FORM_POWER`).

### Head-to-head

The last six meetings between the two clubs adjust expected goals by up to 10%.

### Monte Carlo simulation

**50,000** parallel simulations: for every remaining fixture, goals are drawn
from Poisson distributions at the adjusted xG rates; points are accumulated and
teams ranked. Title, top-four, and relegation probabilities are empirical
frequencies across simulations.

---

## Dependencies

```
pandas
numpy
scipy
requests
tqdm
plotly
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Running the legacy visualisation

The original Plotly charts (points bar chart + Elo line chart) are still available:

```bash
python src/data_processing.py
python src/elo_rating.py
python src/visualization.py
```

---

## License

MIT License. See LICENSE for details.
