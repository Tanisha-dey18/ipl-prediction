"""
STEP 4 of the pipeline: Live chase predictor (bonus)
=====================================================

During the second innings, the scoreboard tells us a lot:
    target 180, score 120/3 after 15 overs  ->  60 needed off 30 balls, 7 wickets left

This script teaches a model to turn that situation into a win chance for the
chasing team, using every ball of every IPL run chase.

What this script does:
  1. Works out the target of every match (first innings total + 1)
  2. Replays each run chase ball by ball, recording the situation after every ball:
     runs left, balls left, wickets left, current and required run rate
  3. Labels every ball with the final answer: did the chasing team win?
  4. Trains Logistic Regression and XGBoost, tests them on the newest seasons
  5. Saves the best one to models/chase_model.pkl and a chart of how the
     win chance moved during the most exciting chase of the test seasons

Run it from anywhere with:
    python src/04_chase_model.py
"""

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
MODELS_DIR = PROJECT_DIR / "models"

# The newest seasons are kept aside for testing.
TEST_SEASONS = 2

# The chase features and chase_state() live in features.py, because the app
# uses them too (same reason as the match features).
from features import CHASE_FEATURES, TOTAL_BALLS


def is_legal(extras_type: pd.Series) -> pd.Series:
    """A wide or a no-ball doesn't count as one of the 6 balls in an over."""
    text = extras_type.fillna("").astype(str).str.lower()
    return ~text.str.contains("wide|noball|no ball|no-ball")


# ---------------------------------------------------------------------------
# Building the ball-by-ball table
# ---------------------------------------------------------------------------
def build_chase_table(matches: pd.DataFrame, deliveries: pd.DataFrame) -> pd.DataFrame:
    # ---- Targets -----------------------------------------------------------
    first_innings = deliveries[deliveries["inning"] == 1]
    targets = first_innings.groupby("match_id")["total_runs"].sum() + 1
    targets.name = "target"

    # Leave out rain-shortened matches (Duckworth-Lewis-Stern): their target
    # and overs were changed, so "runs left" and "balls left" would be wrong.
    m = matches.set_index("id")
    full_matches = pd.Series(True, index=m.index)
    if "method" in m.columns:
        full_matches &= m["method"].isna()
    if "target_overs" in m.columns:
        full_matches &= m["target_overs"].isna() | (m["target_overs"] >= 20)
    good_ids = full_matches[full_matches].index

    # ---- Replay every chase ball by ball -----------------------------------
    chase = deliveries[(deliveries["inning"] == 2) & deliveries["match_id"].isin(good_ids)].copy()
    # Keep the file's own order within each over (a wide is re-bowled with the
    # same ball number), so we use a stable sort on match and over only.
    chase = chase.sort_values(["match_id", "over"], kind="stable")
    chase["legal"] = is_legal(chase["extras_type"]).astype(int)

    # cumsum = running total. After each ball we know the score, wickets and
    # number of legal balls bowled so far in this match.
    by_match = chase.groupby("match_id")
    chase["score"] = by_match["total_runs"].cumsum()
    chase["wickets"] = by_match["is_wicket"].cumsum()
    chase["balls_bowled"] = by_match["legal"].cumsum()

    chase = chase.join(targets, on="match_id")
    chase = chase.join(m[["season", "winner"]], on="match_id")

    # The same calculations as chase_state() in features.py, done for every ball at once.
    # .clip(lower=1) avoids dividing by zero on the very first and last balls.
    chase["runs_left"] = chase["target"] - chase["score"]
    chase["balls_left"] = TOTAL_BALLS - chase["balls_bowled"]
    chase["wickets_left"] = 10 - chase["wickets"]
    chase["crr"] = np.where(chase["balls_bowled"] > 0,
                            chase["score"] * 6 / chase["balls_bowled"].clip(lower=1), 0.0)
    chase["rrr"] = np.where(chase["balls_left"] > 0,
                            chase["runs_left"] * 6 / chase["balls_left"].clip(lower=1), 36.0)

    # Once the target is reached, all out, or the balls are gone, the match is
    # over and there's nothing to predict. The app handles those cases itself.
    live = (chase["runs_left"] > 0) & (chase["balls_left"] > 0) & (chase["wickets_left"] > 0)
    chase = chase[live]

    # The answer: did the chasing (batting) team win this match?
    chase["chaser_won"] = (chase["winner"] == chase["batting_team"]).astype(int)
    return chase[["match_id", "season", "batting_team", "bowling_team",
                  "score", "wickets", "balls_bowled"] + CHASE_FEATURES + ["chaser_won"]]


def make_models() -> dict:
    return {
        # Scaling first: logistic regression works best when all inputs are on
        # similar scales (runs_left is up to ~250, wickets_left only up to 10).
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric="logloss",
        ),
    }


def save_worm_chart(model, table: pd.DataFrame, matches: pd.DataFrame, path: Path):
    """Plot how the chasing team's win chance moved, ball by ball, in the
    most exciting chase of the test seasons (the one where the win chance
    swung the most)."""
    p = pd.Series(model.predict_proba(table[CHASE_FEATURES])[:, 1], index=table.index)
    swing = p.groupby(table["match_id"]).agg(lambda s: s.max() - s.min())
    last_id = swing.idxmax()
    game = table[table["match_id"] == last_id]
    chance = p[game.index] * 100
    overs = game["balls_bowled"] / 6

    chaser = game["batting_team"].iloc[0]
    defender = game["bowling_team"].iloc[0]
    target = int(game["target"].iloc[0])
    winner = matches.set_index("id").loc[last_id, "winner"]

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(x=overs, y=chance, ax=ax, color="#2f6fde", linewidth=2)
    ax.axhline(50, color="black", linestyle="--", linewidth=1)
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 20)
    ax.set_xlabel("Overs into the chase")
    ax.set_ylabel(f"{chaser} win chance (%)")
    ax.set_title(f"{chaser} chasing {target} vs {defender}  (winner: {winner})", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    matches = pd.read_csv(PROCESSED_DIR / "matches_clean.csv", parse_dates=["date"])
    deliveries = pd.read_csv(PROCESSED_DIR / "deliveries_clean.csv", low_memory=False)

    print("Replaying every run chase ball by ball...")
    table = build_chase_table(matches, deliveries)
    order = matches.set_index("id")["date"]
    table = table.assign(_d=table["match_id"].map(order)).sort_values(["_d", "match_id"], kind="stable").drop(columns="_d")
    print(f"  {table['match_id'].nunique():,} chases, {len(table):,} ball situations")

    # ---- Train / test split by season ----------------------------------------
    last_season = int(table["season"].max())
    first_test = last_season - TEST_SEASONS + 1
    train = table[table["season"] < first_test]
    test = table[table["season"] >= first_test]
    print(f"  Training on seasons before {first_test}, testing on {first_test}-{last_season}\n")

    # Two scores per model:
    #   accuracy - how often the side given >50% went on to win
    #   log loss - how good the PERCENTAGES are (lower is better). A model that
    #              says 99% and is wrong gets punished hard. For a win-%
    #              predictor this matters more than accuracy.
    results, trained = {}, {}
    for name, model in make_models().items():
        model.fit(train[CHASE_FEATURES], train["chaser_won"])
        p = model.predict_proba(test[CHASE_FEATURES])[:, 1]
        results[name] = {
            "accuracy": accuracy_score(test["chaser_won"], p > 0.5),
            "log_loss": log_loss(test["chaser_won"], p),
        }
        trained[name] = model

    print("Results on the test seasons (every ball of every chase):")
    for name, r in results.items():
        print(f"   {name:20s} accuracy {r['accuracy'] * 100:5.1f}%   log loss {r['log_loss']:.3f}")

    # Accuracy at different stages of the chase: it should get much better
    # as the chase goes on, because there's less left to happen.
    best_name = min(results, key=lambda n: results[n]["log_loss"])
    p_best = trained[best_name].predict_proba(test[CHASE_FEATURES])[:, 1]
    stage = pd.cut(test["balls_bowled"], [0, 36, 90, 120], labels=["Overs 0-6", "Overs 6-15", "Overs 15-20"])
    by_stage = ((p_best > 0.5) == test["chaser_won"]).groupby(stage, observed=True).mean()
    print(f"\n{best_name} accuracy by stage of the chase:")
    for s, acc in by_stage.items():
        print(f"   {s:12s} {acc * 100:5.1f}%")

    # Chart of an exciting test-season chase, using the model that never saw it.
    MODELS_DIR.mkdir(exist_ok=True)
    save_worm_chart(trained[best_name], test, matches, MODELS_DIR / "chase_worm.png")

    # Retrain the best model on everything for the app.
    final = make_models()[best_name]
    final.fit(table[CHASE_FEATURES], table["chaser_won"])
    joblib.dump(
        {"model": final, "name": best_name, "features": CHASE_FEATURES,
         "test_accuracy": results[best_name]["accuracy"]},
        MODELS_DIR / "chase_model.pkl",
    )
    print(f"\nBest chase model: {best_name}")
    print(f"Saved to {MODELS_DIR / 'chase_model.pkl'}")
    print(f"Chart saved to {MODELS_DIR / 'chase_worm.png'}")


if __name__ == "__main__":
    main()
