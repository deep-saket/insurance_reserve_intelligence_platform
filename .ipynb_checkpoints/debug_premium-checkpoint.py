from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.pipeline import build_datasets
from src.utils.config import ConfigLoader

config = ConfigLoader.load(Path("configs/config.yaml"))

train_dataset, _, _, _ = build_datasets(config)

interest = []
mortality = []
premium = []
sum_assured = []
reserve = []

for record in train_dataset.records:

    reserve.append(record.reserve)

    interest.append(
        record.features[2]
    )

    premium.append(
        record.features[3]
    )

    sum_assured.append(
        record.features[4]
    )

    mortality.append(
        record.features[5]
    )

reserve = np.asarray(reserve)

plt.figure(figsize=(6,5))
plt.scatter(interest,reserve,s=4)
plt.xlabel("Interest Rate")
plt.ylabel("Reserve")
plt.title("Reserve vs Interest")
plt.grid(True)

plt.figure(figsize=(6,5))
plt.scatter(mortality,reserve,s=4)
plt.xlabel("Mortality")
plt.ylabel("Reserve")
plt.title("Reserve vs Mortality")
plt.grid(True)

plt.figure(figsize=(6,5))
plt.scatter(premium,reserve,s=4)
plt.xlabel("Premium")
plt.ylabel("Reserve")
plt.title("Reserve vs Premium")
plt.grid(True)

plt.figure(figsize=(6,5))
plt.scatter(sum_assured,reserve,s=4)
plt.xlabel("Sum Assured")
plt.ylabel("Reserve")
plt.title("Reserve vs Sum Assured")
plt.grid(True)

plt.show()

print("\nFeature ranges\n")

print("Interest")
print(min(interest), max(interest))

print("\nMortality")
print(min(mortality), max(mortality))

print("\nPremium")
print(min(premium), max(premium))

print("\nSum Assured")
print(min(sum_assured), max(sum_assured))

print("\nReserve")
print(min(reserve), max(reserve))