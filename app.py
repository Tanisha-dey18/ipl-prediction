"""
Streamlit app - IPL match winner predictor
==========================================

Two tabs:
  1. Match predictor  - pick two teams, a venue and the toss result, and see
                        each team's chance of winning
  2. Live chase       - enter target, score, overs and wickets during the
                        second innings, and see the chasing team's win chance

Start it from the project folder with:
    streamlit run app.py
Streamlit then opens the app in your web browser (usually http://localhost:8501).

How Streamlit works, in one sentence: it runs this whole file from top to
bottom every time you change something on the page, and every st.something()
call draws one thing on the page.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Let Python find our helper file src/features.py
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_DIR / "src"))
from features import TOTAL_BALLS, chase_state, elo_ratings, make_features  # noqa: E402

MATCHES_FILE = PROJECT_DIR / "data" / "processed" / "matches_clean.csv"
MATCH_MODEL_FILE = PROJECT_DIR / "models" / "match_model.pkl"
CHASE_MODEL_FILE = PROJECT_DIR / "models" / "chase_model.pkl"

st.set_page_config(page_title="IPL Win Predictor", page_icon="🏏", layout="centered")
st.title("🏏 IPL Match Winner Predictor")

# ---------------------------------------------------------------------------
# Friendly check: have the pipeline scripts been run?
# ---------------------------------------------------------------------------
missing = [p for p in (MATCHES_FILE, MATCH_MODEL_FILE, CHASE_MODEL_FILE) if not p.exists()]
if missing:
    st.error(
        "The app needs the cleaned data and trained models, and some are missing:\n\n"
        + "\n".join(f"- `{p.relative_to(PROJECT_DIR)}`" for p in missing)
        + "\n\n**How to fix:** put `matches.csv` and `deliveries.csv` in `data/raw/`, "
        "then run this in the project folder and refresh this page:\n\n"
        "```\npython run_all.py\n```"
    )
    st.stop()  # don't draw the rest of the page


# ---------------------------------------------------------------------------
# Loading data and models
# ---------------------------------------------------------------------------
# Because Streamlit reruns the whole file on every click, we "cache" slow
# things: load them once, then reuse them on every rerun.
@st.cache_data
def load_matches() -> pd.DataFrame:
    matches = pd.read_csv(MATCHES_FILE, parse_dates=["date"])
    return matches.sort_values(["date", "id"]).reset_index(drop=True)


@st.cache_resource
def load_model(path: Path) -> dict:
    # The dictionary saved during training: the model, its name, the feature
    # column order and its test accuracy.
    return joblib.load(path)


def predict_win_chance(bundle, history, team_a, team_b, venue, toss_winner,
                       toss_decision, season) -> tuple[float, dict, dict]:
    """Chance that team_a beats team_b, plus the features from both sides.

    Just like in training, we ask the model twice (A vs B, and B vs A) and
    average, so it doesn't matter which team you pick first.
    """
    model, columns = bundle["model"], bundle["features"]
    feats_a = make_features(history, team_a, team_b, venue, toss_winner, toss_decision, season)
    feats_b = make_features(history, team_b, team_a, venue, toss_winner, toss_decision, season)

    # The model expects a table with the columns in the training order.
    rows = pd.DataFrame([feats_a, feats_b])[columns]
    p_a, p_b = model.predict_proba(rows)[:, 1]
    chance_a = (p_a + (1 - p_b)) / 2
    return chance_a, feats_a, feats_b


matches = load_matches()
match_bundle = load_model(MATCH_MODEL_FILE)
chase_bundle = load_model(CHASE_MODEL_FILE)
latest_season = int(matches["season"].max())

# Only offer teams that played in the latest season (no Deccan Chargers!).
latest = matches[matches["season"] == latest_season]
current_teams = sorted(set(latest["team1"]) | set(latest["team2"]))
# Venues sorted by how many matches were played there, most common first.
venues = matches["venue"].value_counts().index.tolist()

st.caption(f"Based on {len(matches):,} IPL matches from "
           f"{int(matches['season'].min())} to {latest_season}.")

match_tab, chase_tab = st.tabs(["🏟️ Match predictor", "📈 Live chase predictor"])


# ===========================================================================
# TAB 1: Match predictor
# ===========================================================================
with match_tab:
    col1, col2 = st.columns(2)
    team_a = col1.selectbox("Team 1", current_teams, index=0)
    # Team 2's list leaves out Team 1, so you can't pick the same team twice.
    team_b = col2.selectbox("Team 2", [t for t in current_teams if t != team_a], index=0)

    venue = st.selectbox("Venue", venues)

    col3, col4 = st.columns(2)
    toss_winner = col3.radio("Toss won by", [team_a, team_b])
    toss_decision = col4.radio("Toss decision", ["bat", "field"],
                               format_func=lambda d: "Bat first" if d == "bat" else "Field first")

    # We predict "the next season", so the whole history is available and the
    # Elo ratings get their start-of-season adjustment.
    chance_a, feats_a, feats_b = predict_win_chance(
        match_bundle, matches, team_a, team_b, venue, toss_winner, toss_decision,
        season=latest_season + 1,
    )
    chance_b = 1 - chance_a

    st.divider()
    st.subheader("Win probability")
    m1, m2 = st.columns(2)
    m1.metric(team_a, f"{chance_a:.0%}")
    m2.metric(team_b, f"{chance_b:.0%}")
    st.progress(float(chance_a), text=f"{team_a} ◀  ▶ {team_b}")

    favourite = team_a if chance_a >= chance_b else team_b
    if abs(chance_a - chance_b) < 0.06:
        st.info("Too close to call. This one is close to a coin flip.")
    else:
        st.success(f"Predicted winner: **{favourite}** (slight edge).")

    # ---- Why? The numbers behind the prediction -----------------------------
    with st.expander("Why? See the numbers behind this prediction"):
        elo_now = elo_ratings(matches, current_season=latest_season + 1)
        reasons = pd.DataFrame({
            "": ["Head-to-head win rate", "Recent form (last 5)", "Win rate at this venue",
                 "Elo strength rating", "Bats first?"],
            team_a: [f"{feats_a['h2h_rate']:.0%}", f"{feats_a['a_form']:.0%}",
                     f"{feats_a['a_venue_rate']:.0%}", f"{elo_now.get(team_a, 1500):.0f}",
                     "Yes" if feats_a["a_bats_first"] else "No"],
            team_b: [f"{feats_b['h2h_rate']:.0%}", f"{feats_b['a_form']:.0%}",
                     f"{feats_b['a_venue_rate']:.0%}", f"{elo_now.get(team_b, 1500):.0f}",
                     "Yes" if feats_b["a_bats_first"] else "No"],
        }).set_index("")
        st.table(reasons)
        st.caption("A 50% value means there's no history yet (for example, a team "
                   "that has never played at this venue).")

    st.caption(
        f"Model: {match_bundle['name']}. It picked the winner in "
        f"{match_bundle['test_accuracy']:.0%} of past matches when tested season by season, "
        "only slightly better than a coin flip. T20 cricket is very unpredictable "
        "before a ball is bowled, so treat these numbers as a fun guide, not a sure thing."
    )


# ===========================================================================
# TAB 2: Live chase predictor
# ===========================================================================
with chase_tab:
    st.write("Enter the situation during the **second innings** (a full 20-over match).")

    col1, col2 = st.columns(2)
    batting = col1.selectbox("Chasing team", current_teams, key="chase_bat")
    bowling = col2.selectbox("Defending team", [t for t in current_teams if t != batting],
                             key="chase_bowl")

    target = st.number_input("Target (runs needed to win)", min_value=1, max_value=350,
                             value=180, step=1)

    col3, col4, col5 = st.columns(3)
    score = col3.number_input("Current score", min_value=0, max_value=400, value=100, step=1)
    wickets = col4.number_input("Wickets fallen", min_value=0, max_value=10, value=3, step=1)
    overs_text = col5.text_input("Overs completed", value="12.3",
                                 help="Cricket style: 12.3 means 12 overs and 3 balls.")

    # Turn "12.3" into 12 overs + 3 balls = 75 balls.
    try:
        whole, _, part = overs_text.strip().partition(".")
        overs_done, balls_in_over = int(whole or 0), int(part or 0)
        valid_overs = 0 <= overs_done <= 20 and 0 <= balls_in_over <= 5 \
            and not (overs_done == 20 and balls_in_over > 0)
    except ValueError:
        valid_overs = False

    st.divider()

    if not valid_overs:
        st.error("Overs must look like 12.3 (12 overs, 3 balls). "
                 "The part after the dot is 0 to 5, and the maximum is 20.0.")
    else:
        balls_bowled = overs_done * 6 + balls_in_over
        runs_left = target - score
        balls_left = TOTAL_BALLS - balls_bowled

        # Finished matches don't need a model: the rules of cricket decide.
        if runs_left <= 0:
            chase_chance, note = 1.0, f"{batting} have reached the target and won!"
        elif wickets >= 10:
            chase_chance, note = 0.0, f"{batting} are all out. {bowling} won."
        elif balls_left <= 0:
            chase_chance, note = 0.0, f"No balls left. {bowling} won."
        else:
            state = chase_state(target, score, balls_bowled, wickets)
            row = pd.DataFrame([state])[chase_bundle["features"]]
            chase_chance = float(chase_bundle["model"].predict_proba(row)[0, 1])
            note = None

        st.subheader("Win probability")
        m1, m2 = st.columns(2)
        m1.metric(f"{batting} (chasing)", f"{chase_chance:.0%}")
        m2.metric(f"{bowling} (defending)", f"{1 - chase_chance:.0%}")
        st.progress(chase_chance, text=f"{batting} ◀  ▶ {bowling}")

        if note:
            st.success(note)
        else:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Runs needed", runs_left)
            k2.metric("Balls left", balls_left)
            k3.metric("Current run rate", f"{state['crr']:.2f}")
            k4.metric("Required run rate", f"{state['rrr']:.2f}")
            st.caption(f"{batting} need {runs_left} runs from {balls_left} balls "
                       f"with {10 - wickets} wickets in hand.")

    st.caption(
        f"Model: {chase_bundle['name']}, trained on every ball of every IPL run chase. "
        f"On recent seasons it backed the eventual winner {chase_bundle['test_accuracy']:.0%} "
        "of the time (better late in a chase, less sure early on). "
        "It knows the scoreboard only, not who is batting or bowling."
    )
