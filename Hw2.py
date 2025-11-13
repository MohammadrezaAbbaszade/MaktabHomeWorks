import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


########################################
# PART1
########################################

df = pd.read_csv("diabetes.csv")


df = df.fillna(df.mean(numeric_only=True))

print("Missing values per column:")
print(df.isnull().sum())


print("\nCleaned data (first 5 rows):")
print(df.head())

########################################
# PART2
########################################

selected_columns = ["Glucose", "BloodPressure", "BMI", "Age", "Insulin", "Outcome"]
df_selected = df[selected_columns]


corr_matrix = df_selected.corr()
print("Correlation Matrix:")
print(corr_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.xlabel("Features")
plt.ylabel("Features")
plt.title("Correlation Heatmap (Selected Diabetes Features)")
plt.show()


target_corr = corr_matrix["Outcome"].drop("Outcome")


most_correlated_feature = target_corr.idxmax()
most_correlated_value = target_corr.max()

print(f"Feature with highest correlation to Outcome: {most_correlated_feature}")
print(f"Correlation value: {most_correlated_value:.2f}")
