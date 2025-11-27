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
distances = np.sqrt(((X_train - P_test) ** 2).sum(axis=1))


########################################
# PART4
########################################

k = 1
nearest_index = distances.idxmin()
predicted_class_k1 = y_train.iloc[nearest_index]

print("===== K=1 =====")
print("nearest neighbour:", nearest_index)
print("nearest class neighbour:", predicted_class_k1)
print("Real P_test:", y_test_actual)
print("Predicate is true?", predicted_class_k1 == y_test_actual)


########################################
# PART5
########################################


k = 3
nearest_indices_k3 = distances.nsmallest(k).index  # سه همسایه نزدیک
nearest_classes_k3 = y_train.loc[nearest_indices_k3]

predicted_class_k3 = nearest_classes_k3.value_counts().idxmax()

print("===== K=3 =====")
print("three nearest neighbours:", nearest_indices_k3.tolist())
print("these three neighbours classes:", nearest_classes_k3.tolist())
print("Predicated class:", predicted_class_k3)
print("Real P_test:", y_test_actual)
print("Predicate is true?", predicted_class_k3 == y_test_actual)