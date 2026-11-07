"""
Run the whole pipeline with one command
=======================================

    python run_all.py

It runs the four scripts in order. Each one uses the output of the one before:

  1. src/01_clean_data.py     data/raw/*.csv          -> data/processed/*_clean.csv
  2. src/02_features.py       matches_clean.csv       -> data/processed/features.csv
  3. src/03_train_models.py   features.csv            -> models/match_model.pkl
  4. src/04_chase_model.py    *_clean.csv             -> models/chase_model.pkl

Run it again whenever you put a newer dataset in data/raw/.
After it finishes, start the app with:  streamlit run app.py
"""

import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

STEPS = [
    ("Clean the data", "src/01_clean_data.py"),
    ("Build match features", "src/02_features.py"),
    ("Train and compare match models", "src/03_train_models.py"),
    ("Train the live chase model", "src/04_chase_model.py"),
]


def main():
    start = time.time()
    for number, (title, script) in enumerate(STEPS, start=1):
        print("\n" + "=" * 70)
        print(f"STEP {number}/{len(STEPS)}: {title}  ({script})")
        print("=" * 70, flush=True)

        # sys.executable is "the Python running this file", so the scripts use
        # the same Python (and the same installed libraries) as you did.
        result = subprocess.run([sys.executable, str(PROJECT_DIR / script)], cwd=PROJECT_DIR)
        if result.returncode != 0:
            print(f"\nStopped: step {number} ({script}) failed. Read the message above.")
            sys.exit(result.returncode)

    minutes = (time.time() - start) / 60
    print("\n" + "=" * 70)
    print(f"All done in {minutes:.1f} minutes. Start the app with:")
    print("    streamlit run app.py")
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
