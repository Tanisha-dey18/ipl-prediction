"""
STEP 4: Train and compare three models
======================================

What this script does:
  1. Loads the feature table from Step 3
  2. Tests each model season by season ("walk-forward testing"):
       train on 2008-2012 -> predict 2013
       train on 2008-2013 -> predict 2014
       ... and so on up to 2024
     This gives about 770 test matches instead of ~140, so the accuracy
     numbers are much more trustworthy than a single test season.
  3. Compares the models with simple "rules of thumb", so we know whether the
     models are better than a basic guess
  4. Trains the best model on ALL seasons and saves it to
     models/match_model.pkl for the app (Step 5), plus a bar chart to
     models/model_comparison.png

Run it from anywhere with:
    python src/03_train_models.py
"""

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # draw charts to a file, without opening a window
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from features import FEATURE_COLUMNS

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
MODELS_DIR = PROJECT_DIR / "models"

# First season we predict. Earlier seasons are only used for learning, so the
# models always have at least 5 seasons of history behind them.
FIRST_TEST_SEASON = 2013


# ---------------------------------------------------------------------------
# The three models
# ---------------------------------------------------------------------------
# random_state=42 makes the "random" parts repeat exactly each run, so you
# get the same numbers every time. (42 is just a tradition; any number works.)
def make_models() -> dict:
    return {
        # Logistic Regression: draws one straight line between "win" and "lose".
        # Simple, fast, and hard to overfit.
        "Logistic Regression": LogisticRegression(max_iter=1000),

        # Random Forest: 300 decision trees, each trained on a random slice of
        # the data, then they vote. max_depth=5 keeps each tree small, so the
        # forest learns general patterns instead of memorising matches.
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=10, random_state=42
        ),

        # XGBoost: builds trees one after another, each fixing the previous
        # one's mistakes. Very powerful, so we keep it gentle (small trees,
        # small learning rate) because our dataset is small.
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, eval_metric="logloss",
        ),
    }


# ---------------------------------------------------------------------------
# Turning predictions into "right or wrong" per match
# ---------------------------------------------------------------------------
def first_rows(table: pd.DataFrame) -> pd.DataFrame:
    """Each match is in the table twice (A-vs-B and B-vs-A).
    This keeps only the first row of each pair (team1 as Team A)."""
    return table.groupby("match_id").nth(0).set_index("match_id")


def model_correct(model, test: pd.DataFrame) -> pd.Series:
    """For each test match: True if the model picked the winner.

    The model gives a win chance for both rows of a match. We average them
    into one fair number, so the order of the teams can't matter:
        chance A wins = (P(A wins, A listed first) + 1 - P(B wins, B listed first)) / 2
    """
    p = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]  # chance "Team A" wins, per row
    test = test.assign(p=p)
    first = first_rows(test)
    second = test.groupby("match_id").nth(1).set_index("match_id")
    chance_a = (first["p"] + (1 - second["p"])) / 2
    return (chance_a > 0.5) == (first["a_won"] == 1)


def rule_correct(test: pd.DataFrame, rule: str) -> pd.Series:
    """For each test match: True if a simple rule picked the winner."""
    first = first_rows(test)
    if rule == "toss":
        pick_a = first["a_won_toss"] == 1          # toss winner wins
    elif rule == "form":
        pick_a = first["a_form"] >= first["b_form"]  # better recent form wins
    elif rule == "chase":
        pick_a = first["a_bats_first"] == 0        # the chasing team wins
    else:  # "elo"
        pick_a = first["elo_diff"] >= 0            # higher Elo rating wins
    return pick_a == (first["a_won"] == 1)


# ---------------------------------------------------------------------------
# Walk-forward testing
# ---------------------------------------------------------------------------
RULES = {
    "Rule: toss winner wins": "toss",
    "Rule: better form wins": "form",
    "Rule: chasing team wins": "chase",
    "Rule: higher Elo wins": "elo",
}


def walk_forward(table: pd.DataFrame) -> dict:
    """Returns {method name: list of True/False, one per test match}."""
    results = {name: [] for name in list(RULES) + list(make_models())}
    last_season = table["season"].max()

    for season in range(FIRST_TEST_SEASON, last_season + 1):
        train = table[table["season"] < season]   # only the past...
        test = table[table["season"] == season]   # ...to predict this season

        for name, rule in RULES.items():
            results[name].extend(rule_correct(test, rule))

        for name, model in make_models().items():   # fresh, untrained models
            model.fit(train[FEATURE_COLUMNS], train["a_won"])
            results[name].extend(model_correct(model, test))

        print(f"   tested season {season}", end="\r")
    print()
    return results


def save_chart(accuracy: dict, n_matches: int, last_season: int, path: Path):
    # Seaborn works best with a small table: one row per bar.
    chart_data = pd.DataFrame({
        "method": list(accuracy),
        "accuracy": [a * 100 for a in accuracy.values()],
        "kind": ["Simple rule" if n.startswith("Rule") else "ML model" for n in accuracy],
    })

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=chart_data, y="method", x="accuracy", hue="kind",
                palette={"Simple rule": "#9aa5b1", "ML model": "#2f6fde"},
                dodge=False, ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=3)
    ax.axvline(50, color="black", linestyle="--", linewidth=1, label="coin flip (50%)")
    ax.set_xlim(40, chart_data["accuracy"].max() + 6)
    ax.set_xlabel(f"Accuracy on {n_matches} matches, {FIRST_TEST_SEASON}-{last_season}, "
                  "predicted season by season (%)")
    ax.set_ylabel("")
    ax.set_title("IPL match winner: model comparison")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    table = pd.read_csv(PROCESSED_DIR / "features.csv")

    print("Walk-forward testing (each season predicted using only earlier seasons)...")
    results = walk_forward(table)
    n_matches = len(next(iter(results.values())))
    accuracy = {name: pd.Series(correct).mean() for name, correct in results.items()}

    print(f"\nAccuracy over {n_matches} test matches:")
    for name, acc in accuracy.items():
        print(f"   {name:26s} {acc * 100:5.1f}%")

    # How much can these numbers wobble by luck alone? With ~770 matches the
    # "margin of error" is roughly +/- 3.5 points. Differences smaller than
    # that could just be luck, so don't over-read a 1-point win.
    margin = 2 * (0.25 / n_matches) ** 0.5 * 100
    print(f"\n(Margin of error: about +/- {margin:.1f} points)")

    # ---- Pick the best model and retrain it on ALL seasons ------------------
    model_names = list(make_models())
    best_name = max(model_names, key=lambda n: accuracy[n])
    best_model = make_models()[best_name]
    best_model.fit(table[FEATURE_COLUMNS], table["a_won"])

    # Which features does the Random Forest rely on most? (trained on everything)
    rf = make_models()["Random Forest"].fit(table[FEATURE_COLUMNS], table["a_won"])
    importance = pd.Series(rf.feature_importances_, index=FEATURE_COLUMNS)
    print("\nWhat the Random Forest paid most attention to:")
    print((importance.sort_values(ascending=False) * 100).round(1).astype(str).add("%").to_string())

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {"model": best_model, "name": best_name, "features": FEATURE_COLUMNS,
         "test_accuracy": accuracy[best_name]},
        MODELS_DIR / "match_model.pkl",
    )
    save_chart(accuracy, n_matches, int(table["season"].max()), MODELS_DIR / "model_comparison.png")

    print(f"\nBest model: {best_name} ({accuracy[best_name] * 100:.1f}%)")
    print(f"Saved to {MODELS_DIR / 'match_model.pkl'}")
    print(f"Chart saved to {MODELS_DIR / 'model_comparison.png'}")


if __name__ == "__main__":
    main()
