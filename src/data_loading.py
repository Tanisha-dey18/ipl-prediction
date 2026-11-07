"""
Robust data loading (shared code)
=================================

Different versions of the Kaggle IPL dataset use slightly different file names
and column names. For example, one version calls a column "batter" and another
calls it "batsman"; one says "inning", another "innings".

This file:
  1. Finds the matches and deliveries CSV files in data/raw/
  2. Renames columns to ONE standard set of names that the rest of the project uses
  3. Stops with a clear, friendly error if a file or an essential column is missing

If you download a newer season of the dataset, just replace the CSVs in
data/raw/ and run the pipeline again. Nothing else needs to change.
"""

from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"

DATASET_HELP = (
    "\n\nHow to fix this:\n"
    "  1. Download the IPL dataset from Kaggle, for example:\n"
    "     https://www.kaggle.com/datasets/dgsports/ipl-ball-by-ball-2008-to-2025\n"
    "  2. Unzip it.\n"
    f"  3. Put the two CSV files (matches and deliveries) in:\n     {RAW_DIR}\n"
    "  4. Run the pipeline again:  python run_all.py\n"
)


class DatasetError(Exception):
    """Raised when the dataset is missing or doesn't look like IPL data."""


# ---------------------------------------------------------------------------
# Column name aliases
# ---------------------------------------------------------------------------
# Left: the standard name we use. Right: other names seen in dataset versions.
# Column names are first lower-cased and spaces turned into underscores, so
# "Toss Winner" and "toss_winner" already match without an alias.
MATCH_ALIASES = {
    "id": ["id", "match_id", "matchid", "match_number"],
    "season": ["season", "year"],
    "date": ["date", "match_date", "start_date", "dates"],
    "venue": ["venue", "stadium", "ground"],
    "city": ["city"],
    "team1": ["team1", "team_1", "team_a", "home_team"],
    "team2": ["team2", "team_2", "team_b", "away_team"],
    "toss_winner": ["toss_winner", "tosswinner", "toss_won_by"],
    "toss_decision": ["toss_decision", "tossdecision", "toss_choice"],
    "winner": ["winner", "winning_team", "winningteam", "match_winner"],
    "result": ["result", "won_by", "wonby", "result_type"],
    "result_margin": ["result_margin", "margin", "win_margin"],
    "target_runs": ["target_runs", "target"],
    "target_overs": ["target_overs"],
    "method": ["method", "dl_applied", "method_used"],
}
MATCH_REQUIRED = ["id", "venue", "team1", "team2", "toss_winner", "toss_decision", "winner"]

DELIVERY_ALIASES = {
    "match_id": ["match_id", "id", "matchid"],
    "inning": ["inning", "innings", "innings_no", "inning_no"],
    "batting_team": ["batting_team", "battingteam", "team_batting", "batting_side"],
    "bowling_team": ["bowling_team", "bowlingteam", "team_bowling", "bowling_side"],
    "over": ["over", "overs", "over_number"],
    "ball": ["ball", "ballnumber", "ball_number", "ball_in_over"],
    "batter": ["batter", "batsman", "striker"],
    "bowler": ["bowler"],
    "batsman_runs": ["batsman_runs", "batsman_run", "runs_off_bat", "batter_runs", "runs_batter"],
    "extra_runs": ["extra_runs", "extras", "extras_run", "runs_extras"],
    "total_runs": ["total_runs", "total_run", "runs_total", "runs"],
    "extras_type": ["extras_type", "extra_type", "extrastype"],
    "is_wicket": ["is_wicket", "iswicketdelivery", "is_wicket_delivery", "wicket", "is_out"],
    "player_dismissed": ["player_dismissed", "player_out", "playerout", "dismissed_player"],
    "dismissal_kind": ["dismissal_kind", "kind", "wicket_kind", "dismissal_type"],
}
DELIVERY_REQUIRED = ["match_id", "inning", "batting_team", "over", "ball"]


def _standardise_columns(df: pd.DataFrame, aliases: dict, required: list, what: str) -> pd.DataFrame:
    """Rename columns to our standard names, and check the essential ones exist."""
    # "Toss Winner" -> "toss_winner"
    df = df.rename(columns=lambda c: str(c).strip().lower().replace(" ", "_"))

    renames = {}
    for standard, options in aliases.items():
        if standard in df.columns:
            continue  # already has the standard name
        for option in options:
            if option in df.columns and option not in renames:
                renames[option] = standard
                break
    df = df.rename(columns=renames)

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DatasetError(
            f"The {what} file is missing these columns: {missing}\n"
            f"Columns found: {list(df.columns)}\n"
            "If your dataset uses a different name for one of them, add it to the "
            "aliases at the top of src/data_loading.py." + DATASET_HELP
        )
    return df


def _find_csv(kind: str, keywords: list) -> Path:
    """Find the CSV for 'matches' or 'deliveries' in data/raw/.
    First tries the exact name (matches.csv), then any CSV whose name contains
    one of the keywords (e.g. 'IPL_Matches_2008_2025.csv')."""
    if not RAW_DIR.exists():
        raise DatasetError(f"The folder {RAW_DIR} does not exist." + DATASET_HELP)

    exact = RAW_DIR / f"{kind}.csv"
    if exact.exists():
        return exact

    candidates = [p for p in RAW_DIR.glob("*.csv")
                  if any(k in p.name.lower() for k in keywords)]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        names = [p.name for p in candidates]
        raise DatasetError(
            f"Found more than one possible {kind} file in data/raw/: {names}\n"
            f"Rename the one you want to '{kind}.csv' or remove the others."
        )
    found = [p.name for p in RAW_DIR.glob("*.csv")] or "no CSV files"
    raise DatasetError(f"Could not find the {kind} CSV in data/raw/ (found: {found})." + DATASET_HELP)


def load_raw_matches() -> pd.DataFrame:
    path = _find_csv("matches", ["match"])
    print(f"  Reading {path.name}")
    df = pd.read_csv(path, low_memory=False)
    return _standardise_columns(df, MATCH_ALIASES, MATCH_REQUIRED, "matches")


def load_raw_deliveries() -> pd.DataFrame:
    path = _find_csv("deliveries", ["deliver", "ball"])
    print(f"  Reading {path.name}")
    df = pd.read_csv(path, low_memory=False)
    df = _standardise_columns(df, DELIVERY_ALIASES, DELIVERY_REQUIRED, "deliveries")

    # Fill in columns that some versions leave out but can be worked out.
    if "total_runs" not in df.columns:
        if {"batsman_runs", "extra_runs"} <= set(df.columns):
            df["total_runs"] = df["batsman_runs"] + df["extra_runs"]
        else:
            raise DatasetError("The deliveries file has no runs column (total_runs, or "
                               "batsman_runs + extra_runs)." + DATASET_HELP)
    if "is_wicket" not in df.columns:
        if "player_dismissed" in df.columns:
            df["is_wicket"] = df["player_dismissed"].notna().astype(int)
        else:
            raise DatasetError("The deliveries file has no wicket column." + DATASET_HELP)
    if "extras_type" not in df.columns:
        df["extras_type"] = pd.NA

    # Some versions mark wickets as True/False or "Yes"/"No" instead of 1/0.
    df["is_wicket"] = (df["is_wicket"].astype(str).str.lower()
                       .isin(["1", "1.0", "true", "yes", "y"])).astype(int)
    return df
