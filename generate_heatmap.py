"""Generate correlation heatmap for the 11 features used in the KNN pipeline."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv("dataset/Pet_Recommendation_System.csv")

# 9 KNN distance features + 2 hard-filter-only features
columns = [
    'Size', 'EnergyLevel', 'kid_friendliness', 'Vaccinated',
    'shedding', 'MeatConsumption', 'AgeMonths', 'WeightKg',
    'HealthCondition',
    'PetType', 'Gender',
]

corr = df[columns].corr()

fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(
    corr,
    annot=True, fmt=".2f",
    cmap="YlGn",
    linewidths=0.5,
    square=True,
    ax=ax,
)
ax.set_title("Feature Correlation Heatmap — KNN Pipeline (9 features + 2 hard filters)", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

os.makedirs("evaluation_results", exist_ok=True)
plt.savefig("evaluation_results/correlation_heatmap.png", dpi=150)
print("✓ Saved to evaluation_results/correlation_heatmap.png")
