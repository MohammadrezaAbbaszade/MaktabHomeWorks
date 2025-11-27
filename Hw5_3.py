import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


########################################
# PART1
########################################

df = pd.read_csv("iris_dataset.csv")
df = pd.DataFrame(df)

df.columns = ["sepal_length", "sepal_width",
              "petal_length", "petal_width", "species"]

x = df[['sepal_length', 'sepal_width']]
y = df['species']

########################################
# PART2
########################################

P_test = x.iloc[-1]
y_test_actual = y.iloc[-1]

X_train = x.iloc[:-1]
y_train = y.iloc[:-1]

########################################
# PART3
########################################
#without normalize

distances = np.sqrt(((X_train - P_test) ** 2).sum(axis=1))


########################################
# PART4
########################################

k = 3
nearest_indices = distances.nsmallest(k).index
nearest_classes = y_train.loc[nearest_indices]


predicted_class = nearest_classes.value_counts().idxmax()

print("===== K=3 without normalize =====")
print("three nearest neighbours:", nearest_indices.tolist())
print("three nearest neighbours classes:", nearest_classes.tolist())
print("predicated classes:", predicted_class)
print("Real class:", y_test_actual)
print("is predicate true?", predicted_class == y_test_actual)

########################################
# PART5 with normalize
########################################

mean = x.mean()
std = x.std()

X_normalized = (x - mean) / std

X_train_norm = X_normalized.iloc[:-1]
P_test_norm = X_normalized.iloc[-1]

distances_norm = np.sqrt(((X_train_norm - P_test_norm) ** 2).sum(axis=1))
nearest_indices_norm = distances_norm.nsmallest(k).index
nearest_classes_norm = y_train.loc[nearest_indices_norm]
predicted_class_norm = nearest_classes_norm.value_counts().idxmax()

print("===== K=3 without normalize =====")
print("three nearest neighbours:",  nearest_indices_norm.tolist())
print("three nearest neighbours classes:",  nearest_classes_norm.tolist())
print("predicated classes:", predicted_class_norm)
print("Real class:", y_test_actual)
print("is predicate true?", predicted_class_norm == y_test_actual)