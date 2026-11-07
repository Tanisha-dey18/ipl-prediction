"""
STEP 3: Build features for every match
======================================

What this script does:
  1. Loads the clean matches from Step 2 (sorted oldest to newest)
  2. Walks through the matches one by one. For each match it looks ONLY at the
     matches before it, and calculates the features (see features.py)
  3. Writes every match TWICE: once from team1's side and once from team2's side
  4. Saves the result to data/processed/features.csv

Why write every match twice?
  The model answers "does Team A win?". In the app you can pick the teams in
  any order, so the model must not care which team is "A". Showing it each
  match from both sides teaches it exactly that, and it also means half the
  rows have the answer 1 and half have 0, a nice balanced dataset.

Run it from anywhere with:
    python src/02_features.py
"""

from pathlib import Path

import pandas as pd

# Our own helper file, src/features.py
from features import FEATURE_COLUMNS, elo_ratings, make_features

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"


def build_feature_table(matches: pd.DataFrame) -> pd.DataFrame:
    rows = []

    # enumerate gives us the position i (0, 1, 2, ...) and the match itself.
    for i, match in enumerate(matches.itertuples(index=False)):
        # The history is every match BEFORE this one. Because the data is
        # sorted by date, that is simply the first i rows. No peeking!
        history = matches.iloc[:i]

        # Two versions of the match: (team1 as A, team2 as B) and the reverse.
        for team_a, team_b in [(match.team1, match.team2), (match.team2, match.team1)]:
            features = make_features(
                history, team_a, team_b,
                match.venue, match.toss_winner, match.toss_decision,
                match.season,
            )
            rows.append({
                # These extra columns are NOT used by the model. They help us
                # read the table and split it into train/test in Step 4.
                "match_id": match.id,
                "season": match.season,
                "team_a": team_a,
                "team_b": team_b,
                "venue": match.venue,
                **features,   # the ** unpacks the feature dictionary into this row
                # The answer we want the model to learn: did Team A win?
                "a_won": int(match.winner == team_a),
            })

    return pd.DataFrame(rows)


def main():
    print("Loading clean matches...")
    matches = pd.read_csv(PROCESSED_DIR / "matches_clean.csv", parse_dates=["date"])
    # Make sure the order is oldest first (Step 2 already did this, but the
    # "no peeking" rule depends on it, so we double-check).
    matches = matches.sort_values(["date", "id"]).reset_index(drop=True)

    print(f"Building features for {len(matches):,} matches (takes about a minute)...")
    table = build_feature_table(matches)

    out_path = PROCESSED_DIR / "features.csv"
    table.to_csv(out_path, index=False)

    # --- A quick look at what we built ---
    print(f"\nRows: {len(table):,} (2 per match)")
    print("\nFirst 4 rows:")
    print(table.head(4).to_string(index=False))

    # Sanity check: the most recent match, as an example.
    last = table.iloc[-1]
    print(f"\nExample, the last match in the data: {last.team_a} vs {last.team_b} at {last.venue}")
    for col in FEATURE_COLUMNS:
        print(f"   {col:14s} = {last[col]:.2f}")

    # Current Elo ratings after every match, strongest first. A nice check
    # that the ratings make sense for the latest season.
    ratings = elo_ratings(matches)
    active = set(matches.loc[matches["season"] == matches["season"].max(), "team1"])
    print("\nElo ratings at the end of the data (current teams):")
    for team, rating in sorted(ratings.items(), key=lambda kv: -kv[1]):
        if team in active:
            print(f"   {team:30s} {rating:6.0f}")

    # Correlation with the answer: a value above 0 means "when this feature is
    # higher, Team A wins more often". Bigger = more useful (roughly).
    # Values near 0 are normal for single cricket features; a match has a lot
    # of luck in it. The models in Step 4 combine all of them.
    print("\nHow strongly each feature relates to Team A winning:")
    print(table[FEATURE_COLUMNS + ["a_won"]].corr()["a_won"].drop("a_won").round(3).to_string())

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
