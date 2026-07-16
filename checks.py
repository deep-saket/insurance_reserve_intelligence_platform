from src.data.simulator import PolicySimulator

import pandas as pd
import matplotlib.pyplot as plt
from src.utils.config import ConfigLoader

from pathlib import Path

config = ConfigLoader.load(Path("configs/config.yaml"))

plots_dir = Path(
    Path("artifacts") / config.trainer.run_name / "plots"
)

plots_dir.mkdir(
    parents=True,
    exist_ok=True
)

sim = PolicySimulator(
    age_range=(20, 70),
    term_range=(10, 40),
    interest_rate_range=(0.02, 0.10),
    sum_assured_range=(100000, 1000000),
)

policies = sim.generate_random_policies(1000)

rows = []

for p in policies:

    rows.append(
        {
            "age": p.age,
            "term": p.term,
            "premium": p.premium,
            "pricing_interest_rate": p.pricing_interest_rate,
            "scenario_interest_rate": p.scenario_interest_rate,
            "interest_rate": p.scenario_interest_rate,
            "sum_assured": p.sum_assured,

            # mortality at issue
            "mortality": p.mortality_profile.intensities[0],

            "smoker": p.metadata["smoker_status"],
            "health": p.metadata["health_tier"],
            "occupation": p.metadata["occupation_risk"],
            "gender": p.metadata["gender"],
        }
    )

df = pd.DataFrame(rows)

print("\n========== PORTFOLIO SUMMARY ==========")
print(df.describe())

print("\n========== CORRELATIONS ==========")
print(
    df[
        ["age", "premium", "mortality", "sum_assured"]
    ].corr()
)

print("\n========== PREMIUM BY SMOKER ==========")
print(
    df.groupby("smoker")["premium"].mean()
)

print("\n========== MORTALITY BY SMOKER ==========")
print(
    df.groupby("smoker")["mortality"].mean()
)

df["age_band"] = pd.cut(
    df["age"],
    bins=[20,30,40,50,60,70]
)

print("\n========== PREMIUM BY AGE BAND ==========")
print(
    df.groupby("age_band")["premium"].mean()
)

print("\n========== MORTALITY BY AGE BAND ==========")
print(
    df.groupby("age_band")["mortality"].mean()
)

# ------------------------------------------------
# Age vs Mortality
# ------------------------------------------------

plt.figure(figsize=(8,5))

plt.scatter(
    df["age"],
    df["mortality"],
    alpha=0.4
)

plt.xlabel("Age")
plt.ylabel("Mortality")
plt.title("Age vs Mortality")

plt.grid(True)

plt.savefig(
    plots_dir / "age_vs_mortality.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

# ------------------------------------------------
# Age vs Premium
# ------------------------------------------------

plt.figure(figsize=(8,5))

plt.scatter(
    df["age"],
    df["premium"],
    alpha=0.4
)

plt.xlabel("Age")
plt.ylabel("Premium")
plt.title("Age vs Premium")

plt.grid(True)

plt.savefig(
    plots_dir / "age_vs_premium.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

plt.figure(figsize=(6,5))

df.groupby("smoker")["premium"].mean().plot(
    kind="bar"
)

plt.ylabel("Average Premium")
plt.title("Average Premium by Smoker Status")

plt.savefig(
    plots_dir / "premium_by_smoker.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

plt.figure(figsize=(6,5))

df.groupby("smoker")["mortality"].mean().plot(
    kind="bar"
)

plt.ylabel("Average Mortality")
plt.title("Average Mortality by Smoker Status")

plt.savefig(
    plots_dir / "mortality_by_smoker.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print(
    f"\nPlots saved to: {plots_dir}"
)
