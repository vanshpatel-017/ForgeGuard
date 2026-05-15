import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ============================================================
# RESULTS CSV
# ============================================================

CSV_PATH = (

    PROJECT_ROOT /

    "outputs" /

    "metrics" /

    "results.csv"
)

# ============================================================
# LOAD RESULTS
# ============================================================

df = pd.read_csv(CSV_PATH)

print(df.head())

# ============================================================
# PLOT MAP50
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    df["epoch"],
    df["metrics/mAP50(B)"]
)

plt.xlabel("Epoch")

plt.ylabel("mAP50")

plt.title("Training Progress")

plt.grid(True)

plt.show()