"""
Feature helpers (shared code)
=============================

A "feature" is a number that describes a match BEFORE it is played, which the
model can use to guess the winner. We build these features:

  h2h_rate         - how often Team A has beaten Team B in past meetings
  a_form, b_form   - share of the last 5 matches each team won
  a_venue_rate,
  b_venue_rate     - how often each team has won at this ground
  a_won_toss       - 1 if Team A won the toss, else 0
  toss_bat         - 1 if the toss winner chose to bat, 0 if they chose to field
  a_bats_first     - 1 if Team A bats first (worked out from the toss)
  elo_diff         - Team A's Elo strength rating minus Team B's (see below)

The golden rule: every feature is calculated ONLY from matches played before
the match we are describing (the "history"). If we used later matches, the
model would be peeking at the future and its accuracy would be fake.

These functions live in their own file because two scripts use them:
  - 02_features.py  (to build the training data)
  - app.py          (in Step 5, to build features for the match you pick)
"""

import pandas as pd

# The order of the columns the models will be trained on. Keeping it in one
# place means training and the app can never get them in a different order.
FEATURE_COLUMNS = [
    "h2h_rate",
    "a_form",
    "b_form",
    "a_venue_rate",
    "b_venue_rate",
    "a_won_toss",
    "toss_bat",
    "a_bats_first",
    "elo_diff",
]

# How many recent matches count as "recent form".
FORM_MATCHES = 5

# Elo settings (explained in elo_ratings below).
ELO_START = 1500       # every team's rating on day one
ELO_K = 20             # how much one result moves the rating
ELO_CARRY_OVER = 0.75  # share of a team's rating kept into a new season


def win_rate(wins: int, games: int) -> float:
    """Wins divided by games. If there are no games yet, return 0.5
    (a neutral 50/50 guess) instead of dividing by zero."""
    return wins / games if games > 0 else 0.5


def matches_of(history: pd.DataFrame, team: str) -> pd.DataFrame:
    """All matches in the history that this team played in."""
    return history[(history["team1"] == team) | (history["team2"] == team)]


def head_to_head(history: pd.DataFrame, team_a: str, team_b: str) -> float:
    """Share of past A-vs-B matches that Team A won."""
    a_games = matches_of(history, team_a)
    meetings = a_games[(a_games["team1"] == team_b) | (a_games["team2"] == team_b)]
    return win_rate((meetings["winner"] == team_a).sum(), len(meetings))


def recent_form(history: pd.DataFrame, team: str, n: int = FORM_MATCHES) -> float:
    """Share of the team's last n matches that it won.
    .tail(n) takes the last n rows; the history is sorted oldest first,
    so these are the most recent games."""
    last_n = matches_of(history, team).tail(n)
    return win_rate((last_n["winner"] == team).sum(), len(last_n))


def venue_win_rate(history: pd.DataFrame, team: str, venue: str) -> float:
    """Share of the team's matches at this venue that it won."""
    games = matches_of(history, team)
    at_venue = games[games["venue"] == venue]
    return win_rate((at_venue["winner"] == team).sum(), len(at_venue))


def batting_first(team_a: str, team_b: str, toss_winner: str, toss_decision: str) -> str:
    """Who bats first? The toss winner if they chose to bat, otherwise the other team."""
    if toss_decision == "bat":
        return toss_winner
    return team_b if toss_winner == team_a else team_a


def _new_season(ratings: dict) -> dict:
    """At the start of a season, pull every rating 25% of the way back to 1500.
    IPL squads change a lot at auctions, so last year's strength only partly
    carries over."""
    return {team: ELO_CARRY_OVER * r + (1 - ELO_CARRY_OVER) * ELO_START
            for team, r in ratings.items()}


def elo_ratings(history: pd.DataFrame, current_season=None) -> dict:
    """Replay every match in the history and return each team's Elo rating.

    Elo (invented for chess) works like this:
      - every team starts at 1500
      - before a match, the rating gap gives an "expected" result:
        equal ratings = 50% chance, +200 points = about 76% chance
      - after the match, the winner takes points from the loser. Beating a
        stronger team earns MORE points than beating a weaker one.
    So the rating rises and falls with results, and recent results matter most.
    """
    ratings = {}
    season = None
    for match in history.itertuples(index=False):
        if match.season != season:            # a new season has started
            ratings = _new_season(ratings)
            season = match.season

        r1 = ratings.get(match.team1, ELO_START)
        r2 = ratings.get(match.team2, ELO_START)
        expected_1 = 1 / (1 + 10 ** ((r2 - r1) / 400))   # team1's expected win chance
        actual_1 = 1.0 if match.winner == match.team1 else 0.0

        change = ELO_K * (actual_1 - expected_1)  # positive if team1 did better than expected
        ratings[match.team1] = r1 + change
        ratings[match.team2] = r2 - change

    # If the match we are predicting is in a new season, apply the pull-back too.
    if current_season is not None and current_season != season:
        ratings = _new_season(ratings)
    return ratings


def make_features(history: pd.DataFrame, team_a: str, team_b: str,
                  venue: str, toss_winner: str, toss_decision: str,
                  season: int) -> dict:
    """Build all features for one match, seen from Team A's side."""
    elo = elo_ratings(history, current_season=season)
    first = batting_first(team_a, team_b, toss_winner, toss_decision)
    return {
        "h2h_rate": head_to_head(history, team_a, team_b),
        "a_form": recent_form(history, team_a),
        "b_form": recent_form(history, team_b),
        "a_venue_rate": venue_win_rate(history, team_a, venue),
        "b_venue_rate": venue_win_rate(history, team_b, venue),
        "a_won_toss": int(toss_winner == team_a),
        "toss_bat": int(toss_decision == "bat"),
        "a_bats_first": int(first == team_a),
        "elo_diff": elo.get(team_a, ELO_START) - elo.get(team_b, ELO_START),
    }


# ---------------------------------------------------------------------------
# Live chase features (used by 04_chase_model.py and the app)
# ---------------------------------------------------------------------------
TOTAL_BALLS = 120  # 20 overs x 6 balls

CHASE_FEATURES = [
    "runs_left",     # runs still needed
    "balls_left",    # legal balls still to come
    "wickets_left",  # 10 minus wickets fallen
    "target",        # the target itself (a 220 chase is harder than a 140 chase)
    "crr",           # current run rate: runs per over so far
    "rrr",           # required run rate: runs per over still needed
]


def chase_state(target: int, score: int, balls_bowled: int, wickets: int) -> dict:
    """Turn a second-innings scoreboard into the chase model's features.
    Example: target 180, score 120/3 after 15 overs (90 balls)
             -> 60 runs left, 30 balls left, 7 wickets left."""
    runs_left = target - score
    balls_left = TOTAL_BALLS - balls_bowled
    return {
        "runs_left": runs_left,
        "balls_left": balls_left,
        "wickets_left": 10 - wickets,
        "target": target,
        "crr": score * 6 / balls_bowled if balls_bowled > 0 else 0.0,
        "rrr": runs_left * 6 / balls_left if balls_left > 0 else 36.0,
    }
