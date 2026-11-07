"""
STEP 1 of the pipeline: Load and clean the IPL data
===================================================

What this script does:
  1. Finds and loads the matches and deliveries CSVs from data/raw/
     (works with different versions of the Kaggle dataset, see data_loading.py)
  2. Fixes team names that changed over the years (one team = one name)
  3. Fixes venue names that are written in several different ways
  4. Turns messy season labels like "2007/08" into simple years like 2008
  5. Removes matches with no result (washed out by rain - no winner to learn from)
  6. Saves clean copies to data/processed/

Run it from the project folder with:
    python src/01_clean_data.py
"""

import sys
from pathlib import Path

import pandas as pd

from data_loading import DatasetError, load_raw_deliveries, load_raw_matches

# ---------------------------------------------------------------------------
# File locations
# ---------------------------------------------------------------------------
# Path(__file__) is this script's own location. Going up one folder (.parent
# twice) gets us to the project folder, so the script works no matter which
# folder your terminal is in.
PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"


# ---------------------------------------------------------------------------
# 1. Team name fixes
# ---------------------------------------------------------------------------
# Some franchises renamed themselves. If we don't merge the names, the model
# thinks "Delhi Daredevils" and "Delhi Capitals" are two different teams and
# each one only gets half the history.
#
# Left side = old name in the data, right side = the name we will use.
TEAM_NAME_FIXES = {
    "Delhi Daredevils": "Delhi Capitals",                         # renamed in 2019
    "Kings XI Punjab": "Punjab Kings",                            # renamed in 2021
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru", # renamed in 2024
    "Rising Pune Supergiants": "Rising Pune Supergiant",          # spelling changed in 2017
}
# Note: Deccan Chargers and Sunrisers Hyderabad are NOT merged. They were
# different owners and different squads; Sunrisers replaced Deccan in 2013.
# Same for Gujarat Lions (2016-17) and Gujarat Titans (2022 onwards).

MATCH_TEAM_COLUMNS = ["team1", "team2", "toss_winner", "winner"]
DELIVERY_TEAM_COLUMNS = ["batting_team", "bowling_team"]


# ---------------------------------------------------------------------------
# 2. Venue name fixes
# ---------------------------------------------------------------------------
# The same ground appears under several names, for example:
#   "Eden Gardens" and "Eden Gardens, Kolkata"
# Step A (in clean_venue below): cut off everything after the first comma.
# Step B: fix the remaining ones that were renamed or spelled differently.
VENUE_NAME_FIXES = {
    "M.Chinnaswamy Stadium": "M Chinnaswamy Stadium",
    "Feroz Shah Kotla": "Arun Jaitley Stadium",                       # renamed in 2019
    "Feroz Shah Kotla Ground": "Arun Jaitley Stadium",
    "Punjab Cricket Association Stadium": "Punjab Cricket Association IS Bindra Stadium",
    "Sardar Patel Stadium": "Narendra Modi Stadium",                  # Motera, rebuilt and renamed
    "Sheikh Zayed Stadium": "Zayed Cricket Stadium",                  # Abu Dhabi
    "Subrata Roy Sahara Stadium": "Maharashtra Cricket Association Stadium",  # Pune, renamed
}


def clean_team(name):
    """Tidy one team name: remove stray spaces, then apply the rename fixes."""
    if pd.isna(name):
        return name
    name = " ".join(str(name).split())
    return TEAM_NAME_FIXES.get(name, name)


def clean_venue(name) -> str:
    """Turn any spelling of a venue into one standard name."""
    short_name = str(name).split(",")[0].strip()        # Step A: drop ", City"
    return VENUE_NAME_FIXES.get(short_name, short_name)  # Step B: apply fixes


def season_from_label(label) -> int:
    """Backup for when there is no date column: '2007/08' -> 2008, '2019' -> 2019."""
    text = str(label).strip()
    if "/" in text:
        start, end = text.split("/")[:2]
        return int(start[:2] + end[-2:]) if len(end) == 2 else int(end)
    return int(float(text))


# ---------------------------------------------------------------------------
# The main cleaning steps
# ---------------------------------------------------------------------------
def clean_matches(matches: pd.DataFrame) -> pd.DataFrame:
    df = matches.copy()

    # Fix team names in every team column.
    for col in MATCH_TEAM_COLUMNS:
        df[col] = df[col].apply(clean_team)

    df["venue"] = df["venue"].apply(clean_venue)
    # "Field", "field " and "FIELD" should all be the same thing.
    df["toss_decision"] = df["toss_decision"].astype(str).str.strip().str.lower()

    # Dates: different versions write dates differently, so try a couple of ways.
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date"].isna().mean() > 0.5:   # mostly failed? try day-first (31/12/2020)
            df["date"] = pd.to_datetime(matches["date"], errors="coerce", dayfirst=True)

    # Season: the date's year is the most reliable. The raw "season" labels
    # are messy ("2007/08", "2020/21"), so they're only a backup.
    if "date" in df.columns and df["date"].notna().all():
        df["season"] = df["date"].dt.year
    elif "season" in df.columns:
        df["season"] = df["season"].apply(season_from_label)
    else:
        raise DatasetError("The matches file needs a 'date' or a 'season' column.")
    if "date" not in df.columns or df["date"].isna().any():
        # No usable dates: fall back to season order, then match id order.
        df["date"] = pd.to_datetime(df["season"].astype(str) + "-01-01")

    # Drop matches with no winner (rain washouts). Some versions write "NA" or
    # "No result" instead of leaving it empty, so we also drop any winner that
    # isn't one of the two teams. Tied matches are kept: the super over decided
    # a winner, and that winner is in the 'winner' column.
    before = len(df)
    df = df[(df["winner"] == df["team1"]) | (df["winner"] == df["team2"])]
    print(f"  Removed {before - len(df)} matches with no result")

    # Keep only the columns we need later (if this dataset version has them).
    keep = [
        "id", "season", "date", "venue", "city",
        "team1", "team2", "toss_winner", "toss_decision",
        "winner", "result", "result_margin", "target_runs", "target_overs", "method",
    ]
    df = df[[c for c in keep if c in df.columns]]

    # Sort by date, oldest first. This matters a lot later: features like
    # "recent form" must only look at matches played BEFORE each match.
    df = df.sort_values(["date", "id"]).reset_index(drop=True)
    return df


def clean_deliveries(deliveries: pd.DataFrame, match_ids) -> pd.DataFrame:
    df = deliveries.copy()

    for col in DELIVERY_TEAM_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(clean_team)

    # Keep only balls from matches we kept (so washed-out games disappear too).
    df = df[df["match_id"].isin(match_ids)]
    return df.reset_index(drop=True)


def main():
    print("Loading raw data...")
    matches = load_raw_matches()
    deliveries = load_raw_deliveries()
    print(f"  matches:    {len(matches):,} rows")
    print(f"  deliveries: {len(deliveries):,} rows")

    print("\nCleaning matches...")
    matches_clean = clean_matches(matches)

    print("Cleaning deliveries...")
    deliveries_clean = clean_deliveries(deliveries, matches_clean["id"])
    if deliveries_clean.empty:
        raise DatasetError(
            "None of the deliveries matched a match id from the matches file. "
            "Check that both CSVs come from the same dataset download."
        )

    # Save. index=False stops pandas from adding an extra unnamed column.
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    matches_clean.to_csv(PROCESSED_DIR / "matches_clean.csv", index=False)
    deliveries_clean.to_csv(PROCESSED_DIR / "deliveries_clean.csv", index=False)

    # A quick summary so you can see the cleaning worked.
    teams = sorted(set(matches_clean["team1"]) | set(matches_clean["team2"]))
    print("\n--- Summary ---")
    print(f"Matches kept:  {len(matches_clean):,}")
    print(f"Balls kept:    {len(deliveries_clean):,}")
    print(f"Seasons:       {matches_clean['season'].min()} to {matches_clean['season'].max()}")
    print(f"Venues:        {matches_clean['venue'].nunique()} (was {matches['venue'].nunique()})")
    print(f"Teams ({len(teams)}):")
    for team in teams:
        print(f"   - {team}")
    print(f"\nSaved to {PROCESSED_DIR}")


# This line means: only run main() when you run this file directly,
# not when another script imports something from it.
if __name__ == "__main__":
    try:
        main()
    except DatasetError as error:
        print(f"\nDATASET PROBLEM: {error}")
        sys.exit(1)
