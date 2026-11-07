# 🏏 IPL Match Winner Predictor

A beginner-friendly machine learning project that:

- predicts the winner of an IPL match and shows each team's win probability,
  using head-to-head record, recent form, venue record, toss and Elo strength
- compares **Logistic Regression**, **Random Forest** and **XGBoost**
- has a **Streamlit** web app with a match predictor and a **live chase predictor**
  (target, score, overs, wickets → win %)

Every script is commented line by line so you can learn from it.

---

## 1. Setup (once)

Needs Python 3.11 or newer (3.13 works too). In PowerShell, inside this folder:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell says *"running scripts is disabled"*, skip the `Activate.ps1` line
and put `.\venv\Scripts\` in front of every `python` and `streamlit` command below.

## 2. Get the data

1. Download an IPL ball-by-ball dataset from Kaggle (free account needed), for example
   <https://www.kaggle.com/datasets/dgsports/ipl-ball-by-ball-2008-to-2025>
   or "IPL Complete Dataset (2008-2024)".
2. Unzip it and put the two CSV files in `data/raw/`:
   - the matches file (one row per match), e.g. `matches.csv`
   - the deliveries file (one row per ball), e.g. `deliveries.csv`

Different dataset versions use different column names (`batter` vs `batsman`,
`inning` vs `innings`, ...). `src/data_loading.py` handles the common ones and
gives a clear error if something is missing. For a newer season, just replace
the CSVs and run step 3 again.

## 3. Train everything

```powershell
python run_all.py
```

Takes about 2 minutes. It runs:

| Script | What it does | Output |
|---|---|---|
| `src/01_clean_data.py` | Fixes renamed teams (Delhi Daredevils → Delhi Capitals, ...) and venue spellings, drops no-result games | `data/processed/*_clean.csv` |
| `src/02_features.py` | Builds features for every match using **only earlier matches** | `data/processed/features.csv` |
| `src/03_train_models.py` | Tests the 3 models season by season, saves the best | `models/match_model.pkl`, `models/model_comparison.png` |
| `src/04_chase_model.py` | Trains the live chase model on every ball of every run chase | `models/chase_model.pkl`, `models/chase_worm.png` |

## 4. Run the app

```powershell
streamlit run app.py
```

It opens in your browser at <http://localhost:8501>. Stop it with **Ctrl + C**.

---

## Results (2008–2024 dataset)

**Match winner (before the match starts)**, tested on 769 matches, 2013–2024,
each season predicted using only earlier seasons:

| Method | Accuracy |
|---|---|
| Rule: toss winner wins | 50.8% |
| Rule: chasing team wins | 54.1% |
| Logistic Regression | 51.2% |
| **Random Forest** (used in the app) | **53.1%** |
| XGBoost | 51.6% |

That's only a little better than a coin flip, and it's an honest result. T20
cricket is very hard to predict before a ball is bowled, and squads change every
year at the auction. Projects claiming 80–90% here usually leak future results
into their features by mistake. This project avoids that: every feature for a
match uses only matches played before it.

**Live chase**, tested on 2023–2024: the model backs the eventual winner on
**~79%** of balls overall, and **~88%** in the last 5 overs.

## Project structure

```
ipl-predictor/
├── data/
│   ├── raw/              ← put the Kaggle CSVs here
│   └── processed/        ← created by the scripts
├── models/               ← trained models and charts
├── src/
│   ├── data_loading.py   ← finds the CSVs, handles different column names
│   ├── features.py       ← feature helpers shared by training and the app
│   ├── 01_clean_data.py
│   ├── 02_features.py
│   ├── 03_train_models.py
│   └── 04_chase_model.py
├── app.py                ← Streamlit app
├── run_all.py            ← runs all four scripts in order
└── requirements.txt
```
