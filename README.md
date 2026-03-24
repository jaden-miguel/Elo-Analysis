# Premier League Title-Race Prediction (2022-2026)

This project fetches live Premier League match data across multiple seasons,
fits a statistically rigorous prediction model, and produces an interactive
HTML dashboard that forecasts which team will win the current season's title.

---

## Features

- Live match data from [openfootball](https://github.com/openfootball/football.json) (2022-23 through the current season)
- Dixon-Coles time-weighted Poisson model — the same family of models used by professional football analytics teams
- Separate attack and defensive strength ratings for every club
- Exponential time decay so recent results count more than older ones
- Low-score correction (rho) fixes the systematic under-prediction of 0-0 and 1-0 draws
- Current-season form index (last 6 matches) shifts expected goal rates up or down
- 10,000 Monte Carlo simulations of remaining fixtures to produce title-win probabilities
- Elo rating tracker across all seasons for trend visualisation
- Self-contained HTML dashboard: probability bars with club badges, form badges (W/D/L), current standings, and an interactive Elo trajectory chart
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
1. Fit the Dixon-Coles model to all completed matches
2. Run 10,000 Monte Carlo simulations of remaining fixtures
3. Generate `data/prediction_dashboard.html`
4. Open the dashboard automatically in your default browser

---

## The Prediction Model

### Why Dixon-Coles over Elo?

Standard Elo tracks only win/draw/loss. The Dixon-Coles model instead models
the number of goals each team scores, which contains significantly more
information.

Every team has two fitted parameters:

| Parameter | Meaning |
|-----------|---------|
| `attack[team]` | Log attacking strength (positive = more goals scored than average) |
| `defense[team]` | Log defensive quality (positive = fewer goals conceded than average) |

Global parameters: `home_adv` (home-field advantage) and `rho` (low-score
correlation correction).

### Time weighting

Each historical match is weighted by `exp(-0.003 * days_ago)`, which gives
roughly half-weight to matches played more than 8 months ago. This ensures
the model reflects current team quality rather than being dominated by old data.

### Form adjustment

The model computes each team's points-per-game rate over their last 6 matches
in the current season. Teams in good form have their expected goals shifted up
by up to 20%; teams on a poor run are shifted down by the same amount.

### Monte Carlo simulation

For each of 10,000 simulations, every remaining fixture is resolved by:
1. Computing expected goals for home and away using fitted parameters and form
2. Sampling actual goals from a Poisson distribution
3. Applying the Dixon-Coles low-score correction for 0-0, 0-1, 1-0, 1-1 outcomes
4. Accumulating points across all teams

The fraction of simulations in which each team finishes top is their title-win probability.

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
