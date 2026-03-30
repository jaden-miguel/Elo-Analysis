"""
Premier League Player Database
===============================
Curated database of PL players with stats and scouting attributes.
Used for the scout section and goal scorer predictions.
"""


PL_PLAYERS = [
    # Rising stars
    {"name": "Lamine Yamal",     "team": "Brighton",      "pos": "RW", "age": 17, "nat": "ESP", "goals": 9,  "assists": 11, "apps": 28, "mins": 2100, "value_m": 120, "potential": 95, "category": "rising_star"},
    {"name": "Kobbie Mainoo",    "team": "Man Utd",       "pos": "CM", "age": 19, "nat": "ENG", "goals": 3,  "assists": 5,  "apps": 30, "mins": 2400, "value_m": 55,  "potential": 88, "category": "rising_star"},
    {"name": "Evan Ferguson",    "team": "Brighton",      "pos": "ST", "age": 20, "nat": "IRL", "goals": 7,  "assists": 2,  "apps": 25, "mins": 1600, "value_m": 45,  "potential": 87, "category": "rising_star"},
    {"name": "Rico Lewis",       "team": "Man City",      "pos": "RB", "age": 20, "nat": "ENG", "goals": 2,  "assists": 6,  "apps": 28, "mins": 2000, "value_m": 40,  "potential": 86, "category": "rising_star"},
    {"name": "Alejandro Garnacho","team": "Man Utd",      "pos": "LW", "age": 20, "nat": "ARG", "goals": 8,  "assists": 5,  "apps": 32, "mins": 2200, "value_m": 50,  "potential": 87, "category": "rising_star"},
    {"name": "Curtis Jones",     "team": "Liverpool",     "pos": "CM", "age": 23, "nat": "ENG", "goals": 5,  "assists": 4,  "apps": 27, "mins": 1900, "value_m": 35,  "potential": 84, "category": "rising_star"},
    {"name": "Morgan Gibbs-White","team": "Nottm Forest", "pos": "AM", "age": 24, "nat": "ENG", "goals": 6,  "assists": 8,  "apps": 30, "mins": 2500, "value_m": 40,  "potential": 83, "category": "rising_star"},
    {"name": "Ethan Nwaneri",    "team": "Arsenal",       "pos": "AM", "age": 17, "nat": "ENG", "goals": 4,  "assists": 3,  "apps": 18, "mins": 900,  "value_m": 25,  "potential": 90, "category": "rising_star"},
    {"name": "Tyler Dibling",    "team": "Sunderland",    "pos": "RW", "age": 18, "nat": "ENG", "goals": 5,  "assists": 4,  "apps": 22, "mins": 1500, "value_m": 20,  "potential": 85, "category": "rising_star"},
    {"name": "Archie Gray",      "team": "Tottenham",     "pos": "CM", "age": 18, "nat": "ENG", "goals": 1,  "assists": 3,  "apps": 24, "mins": 1700, "value_m": 30,  "potential": 86, "category": "rising_star"},
    {"name": "Jobe Bellingham",  "team": "Sunderland",    "pos": "CM", "age": 19, "nat": "ENG", "goals": 6,  "assists": 5,  "apps": 26, "mins": 2000, "value_m": 25,  "potential": 84, "category": "rising_star"},
    {"name": "Harvey Elliott",   "team": "Liverpool",     "pos": "AM", "age": 21, "nat": "ENG", "goals": 4,  "assists": 7,  "apps": 25, "mins": 1700, "value_m": 35,  "potential": 85, "category": "rising_star"},
    {"name": "Carlos Baleba",    "team": "Brighton",      "pos": "CDM","age": 20, "nat": "CMR", "goals": 1,  "assists": 2,  "apps": 22, "mins": 1500, "value_m": 25,  "potential": 83, "category": "rising_star"},
    {"name": "Sam Iling-Junior", "team": "Aston Villa",   "pos": "LW", "age": 21, "nat": "ENG", "goals": 3,  "assists": 4,  "apps": 20, "mins": 1200, "value_m": 15,  "potential": 80, "category": "rising_star"},
    {"name": "Crysencio Summerville","team": "Leeds",     "pos": "LW", "age": 22, "nat": "NED", "goals": 10, "assists": 6,  "apps": 30, "mins": 2400, "value_m": 30,  "potential": 83, "category": "rising_star"},
    # Golden boot contenders
    {"name": "Erling Haaland",   "team": "Man City",      "pos": "ST", "age": 24, "nat": "NOR", "goals": 22, "assists": 4,  "apps": 28, "mins": 2400, "value_m": 170, "potential": 95, "category": "golden_boot"},
    {"name": "Mohamed Salah",    "team": "Liverpool",     "pos": "RW", "age": 32, "nat": "EGY", "goals": 19, "assists": 13, "apps": 29, "mins": 2500, "value_m": 65,  "potential": 89, "category": "golden_boot"},
    {"name": "Alexander Isak",   "team": "Newcastle",     "pos": "ST", "age": 25, "nat": "SWE", "goals": 17, "assists": 5,  "apps": 27, "mins": 2300, "value_m": 90,  "potential": 89, "category": "golden_boot"},
    {"name": "Bukayo Saka",      "team": "Arsenal",       "pos": "RW", "age": 23, "nat": "ENG", "goals": 14, "assists": 11, "apps": 29, "mins": 2500, "value_m": 120, "potential": 92, "category": "golden_boot"},
    {"name": "Cole Palmer",      "team": "Chelsea",       "pos": "AM", "age": 23, "nat": "ENG", "goals": 16, "assists": 9,  "apps": 28, "mins": 2400, "value_m": 100, "potential": 91, "category": "golden_boot"},
    {"name": "Son Heung-min",    "team": "Tottenham",     "pos": "LW", "age": 32, "nat": "KOR", "goals": 13, "assists": 7,  "apps": 29, "mins": 2400, "value_m": 40,  "potential": 86, "category": "golden_boot"},
    {"name": "Ollie Watkins",    "team": "Aston Villa",   "pos": "ST", "age": 28, "nat": "ENG", "goals": 12, "assists": 8,  "apps": 30, "mins": 2600, "value_m": 60,  "potential": 85, "category": "golden_boot"},
    {"name": "Chris Wood",       "team": "Nottm Forest",  "pos": "ST", "age": 32, "nat": "NZL", "goals": 14, "assists": 3,  "apps": 28, "mins": 2300, "value_m": 8,   "potential": 78, "category": "golden_boot"},
    {"name": "Bryan Mbeumo",     "team": "Brentford",     "pos": "RW", "age": 25, "nat": "CMR", "goals": 13, "assists": 6,  "apps": 29, "mins": 2400, "value_m": 45,  "potential": 84, "category": "golden_boot"},
    {"name": "Jean-Philippe Mateta","team":"Crystal Palace","pos":"ST","age": 27, "nat": "FRA", "goals": 11, "assists": 2,  "apps": 26, "mins": 2100, "value_m": 30,  "potential": 80, "category": "golden_boot"},
    # Key players
    {"name": "Martin Odegaard",  "team": "Arsenal",       "pos": "AM", "age": 25, "nat": "NOR", "goals": 8,  "assists": 10, "apps": 25, "mins": 2100, "value_m": 110, "potential": 91, "category": "key_player"},
    {"name": "Kevin De Bruyne",  "team": "Man City",      "pos": "AM", "age": 33, "nat": "BEL", "goals": 4,  "assists": 12, "apps": 22, "mins": 1700, "value_m": 40,  "potential": 90, "category": "key_player"},
    {"name": "Bruno Fernandes",  "team": "Man Utd",       "pos": "AM", "age": 30, "nat": "POR", "goals": 7,  "assists": 8,  "apps": 30, "mins": 2600, "value_m": 55,  "potential": 87, "category": "key_player"},
    {"name": "Phil Foden",       "team": "Man City",      "pos": "LW", "age": 24, "nat": "ENG", "goals": 9,  "assists": 7,  "apps": 26, "mins": 2000, "value_m": 100, "potential": 91, "category": "key_player"},
    {"name": "Declan Rice",      "team": "Arsenal",       "pos": "CDM","age": 26, "nat": "ENG", "goals": 5,  "assists": 6,  "apps": 29, "mins": 2500, "value_m": 100, "potential": 89, "category": "key_player"},
    {"name": "William Saliba",   "team": "Arsenal",       "pos": "CB", "age": 23, "nat": "FRA", "goals": 2,  "assists": 1,  "apps": 28, "mins": 2500, "value_m": 90,  "potential": 90, "category": "key_player"},
    {"name": "Virgil van Dijk",  "team": "Liverpool",     "pos": "CB", "age": 33, "nat": "NED", "goals": 3,  "assists": 1,  "apps": 28, "mins": 2500, "value_m": 30,  "potential": 87, "category": "key_player"},
    {"name": "Pedro Neto",       "team": "Chelsea",       "pos": "RW", "age": 24, "nat": "POR", "goals": 5,  "assists": 8,  "apps": 26, "mins": 1900, "value_m": 55,  "potential": 85, "category": "key_player"},
    {"name": "Eberechi Eze",     "team": "Crystal Palace","pos": "AM", "age": 26, "nat": "ENG", "goals": 7,  "assists": 5,  "apps": 27, "mins": 2200, "value_m": 50,  "potential": 85, "category": "key_player"},
    {"name": "James Maddison",   "team": "Tottenham",     "pos": "AM", "age": 27, "nat": "ENG", "goals": 6,  "assists": 9,  "apps": 28, "mins": 2200, "value_m": 45,  "potential": 84, "category": "key_player"},
    {"name": "Dominic Solanke",  "team": "Tottenham",     "pos": "ST", "age": 27, "nat": "ENG", "goals": 10, "assists": 4,  "apps": 28, "mins": 2300, "value_m": 50,  "potential": 83, "category": "key_player"},
    {"name": "Matheus Cunha",    "team": "Wolves",        "pos": "ST", "age": 25, "nat": "BRA", "goals": 11, "assists": 6,  "apps": 29, "mins": 2400, "value_m": 45,  "potential": 84, "category": "key_player"},
    {"name": "Jarrod Bowen",     "team": "West Ham",      "pos": "RW", "age": 27, "nat": "ENG", "goals": 8,  "assists": 7,  "apps": 29, "mins": 2400, "value_m": 45,  "potential": 83, "category": "key_player"},
    {"name": "Antoine Semenyo",  "team": "Bournemouth",   "pos": "RW", "age": 24, "nat": "GHA", "goals": 8,  "assists": 5,  "apps": 28, "mins": 2200, "value_m": 30,  "potential": 82, "category": "key_player"},
    {"name": "Callum Wilson",    "team": "Newcastle",     "pos": "ST", "age": 32, "nat": "ENG", "goals": 6,  "assists": 2,  "apps": 18, "mins": 1200, "value_m": 12,  "potential": 78, "category": "key_player"},
    {"name": "Yoane Wissa",      "team": "Brentford",     "pos": "ST", "age": 28, "nat": "COD", "goals": 10, "assists": 3,  "apps": 27, "mins": 2100, "value_m": 25,  "potential": 80, "category": "key_player"},
    {"name": "Joao Pedro",       "team": "Brighton",      "pos": "ST", "age": 23, "nat": "BRA", "goals": 8,  "assists": 5,  "apps": 26, "mins": 2000, "value_m": 35,  "potential": 84, "category": "key_player"},
    {"name": "Michael Olise",    "team": "Chelsea",       "pos": "RW", "age": 23, "nat": "FRA", "goals": 7,  "assists": 6,  "apps": 24, "mins": 1800, "value_m": 60,  "potential": 88, "category": "key_player"},
    {"name": "Jack Grealish",    "team": "Man City",      "pos": "LW", "age": 29, "nat": "ENG", "goals": 3,  "assists": 5,  "apps": 22, "mins": 1500, "value_m": 35,  "potential": 83, "category": "key_player"},
    {"name": "Hwang Hee-chan",   "team": "Wolves",        "pos": "ST", "age": 28, "nat": "KOR", "goals": 7,  "assists": 3,  "apps": 26, "mins": 2000, "value_m": 20,  "potential": 80, "category": "key_player"},
    {"name": "Brennan Johnson",  "team": "Tottenham",     "pos": "RW", "age": 24, "nat": "WAL", "goals": 6,  "assists": 4,  "apps": 27, "mins": 1900, "value_m": 35,  "potential": 82, "category": "key_player"},
    {"name": "Kaoru Mitoma",     "team": "Brighton",      "pos": "LW", "age": 27, "nat": "JPN", "goals": 5,  "assists": 6,  "apps": 24, "mins": 1700, "value_m": 35,  "potential": 82, "category": "key_player"},
    {"name": "Nicolas Jackson",  "team": "Chelsea",       "pos": "ST", "age": 23, "nat": "SEN", "goals": 12, "assists": 5,  "apps": 29, "mins": 2300, "value_m": 45,  "potential": 84, "category": "golden_boot"},
    {"name": "Danny Welbeck",    "team": "Brighton",      "pos": "ST", "age": 33, "nat": "ENG", "goals": 7,  "assists": 3,  "apps": 24, "mins": 1600, "value_m": 5,   "potential": 74, "category": "key_player"},
]


def compute_breakout_scores(players: list, elo_ratings: dict,
                            form_scores: dict, sname_fn) -> list:
    """
    Compute a 0-100 breakout score for each player based on:
    age, goal contributions, playing time, team Elo, potential, team form.
    """
    display_elos = {sname_fn(t): e for t, e in elo_ratings.items()}
    display_form = {sname_fn(t): f for t, f in form_scores.items()}

    scored = []
    for p in players:
        team = p["team"]
        age = p["age"]
        goals = p["goals"]
        assists = p["assists"]
        apps = p["apps"]
        mins = p["mins"]
        potential = p.get("potential", 75)
        value_m = p.get("value_m", 10)

        age_factor = max(0, (25 - age) / 10) if age <= 25 else 0
        contrib = (goals * 1.5 + assists) / max(apps, 1)
        minutes_factor = min(mins / 2500, 1.0)
        elo = display_elos.get(team, 1500)
        elo_factor = min((elo - 1400) / 300, 1.0) if elo > 1400 else 0
        pot_factor = (potential - 70) / 25
        form = display_form.get(team, 1.0)
        form_factor = min(form / 1.5, 1.0) if isinstance(form, (int, float)) else 0.5

        breakout = int(
            age_factor * 25
            + contrib * 15
            + minutes_factor * 15
            + elo_factor * 15
            + pot_factor * 20
            + form_factor * 10
        )
        breakout = max(0, min(100, breakout))

        scored.append({
            **p,
            "breakout": breakout,
        })

    scored.sort(key=lambda x: x["breakout"], reverse=True)
    return scored
