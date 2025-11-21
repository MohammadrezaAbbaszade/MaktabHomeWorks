import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


########################################
# PART1
########################################

df = pd.read_csv("iris.data",header=None)
df = pd.DataFrame(df)

df.columns = ["sepal_length", "sepal_width",
              "petal_length", "petal_width", "species"]

# print(df.head())

df["petal_ratio"] = df["petal_length"] / df["petal_width"]

print(df["petal_ratio"])
# print(df.info())
# print(df.isnull().sum())


# df = df.fillna(df.mean(numeric_only=True))
#
# print(df.corr(numeric_only=True))


##############################################
# HeatMap
##############################################
plt.figure(figsize=(8, 5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="Blues")
plt.title("Correlation Heatmap")
plt.show()

##############################################
# Pairplot
##############################################

sns.pairplot(df, hue="species")
plt.show()


##############################################
# BoxPlot
##############################################

plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x="species", y="petal_ratio")
plt.title("Petal Ratio Distribution by Species")
plt.show()

##############################################
# Scatter Plot
##############################################

plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x="petal_length", y="petal_ratio", hue="species")
plt.title("Petal Length vs Petal Ratio")
plt.show()

