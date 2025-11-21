import pandas as pd
from collections import Counter
import re
import matplotlib.pyplot as plt
import seaborn as sns

########################################
# PART1
########################################
df = pd.read_csv("reviews.csv")


print(df.columns)


print(df.head())


if "ReviewText" in df.columns:
    print("Column 'ReviewText' is present.")
else:
    print("Column 'ReviewText' is missing.")


########################################
# PART2
########################################

print("Number of NaN in ReviewText:", df["ReviewText"].isna().sum())


df_cleaned = df.dropna(subset=["ReviewText"])


df["ReviewText"].fillna("", inplace=True)

print("Number of NaN after cleaning:", df["ReviewText"].isna().sum())

########################################
# PART3
########################################

df["TextLength"] = df["ReviewText"].apply(len)

print(df[["ReviewText", "TextLength"]].head())

df["WordCount"] = df["ReviewText"].apply(lambda x: len(x.split()))


print(df[["ReviewText", "WordCount"]].head())

########################################
# PART4
########################################

all_text = " ".join(df["ReviewText"])

clean_text = re.sub(r'[^\w\s]', '', all_text.lower())

tokens = clean_text.split()
word_counts = Counter(tokens)
top_10 = word_counts.most_common(10)
print(top_10)

########################################
# PART5
########################################
words, counts = zip(*top_10)

plt.figure(figsize=(10,6))
sns.barplot(x=list(words), y=list(counts), palette="viridis")
plt.title("Top 10 Most Frequent Words in Reviews")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.show()