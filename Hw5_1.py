import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#
# ################################################################################
# # Question1
# ################################################################################
#
# ########################################
# # PART1
# ########################################
#
# df = pd.read_csv("iris_dataset.csv")
# df = pd.DataFrame(df)
# print(df.head())
#
# grouped = df.groupby("target")
#
# for feature in df.columns[:-1]:
#     print(f"Feature: {feature}")
#     for species, group in grouped:
#         mean = group[feature].mean()
#         std_dev = group[feature].std()
#         range_value = group[feature].max() - group[feature].min()
#         print(f"  {species}: Mean = {mean}, Std Dev = {std_dev}, Range = {range_value}")
#         print("\n")
#
#
# ########################################
# # PART2
# ########################################
# correlation_matrix = df.iloc[:, :-1].corr()
#
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
# plt.title('Correlation Heatmap')
# plt.show()
#
#
#
#
# ################################################################################
# # Question2
# ################################################################################
#
# ########################################
# # PART1
# ########################################
#
# df = pd.read_csv("iris_dataset.csv")
# df = pd.DataFrame(df)
# print(df.head())
#
# x = df[['sepal length (cm)', 'sepal width (cm)']]
# y = df['target']
#
# ########################################
# # PART2
# ########################################
#
# P_test = x.iloc[-1]
# y_test_actual = y.iloc[-1]
#
# X_train = x.iloc[:-1]
# y_train = y.iloc[:-1]
#
# ########################################
# # PART3
# ########################################
# #without normalize
#
# distances = np.sqrt(((X_train - P_test) ** 2).sum(axis=1))
#
#
# ########################################
# # PART4
# ########################################
#
# k = 3
# nearest_indices = distances.nsmallest(k).index
# nearest_classes = y_train.loc[nearest_indices]
#
#
# predicted_class = nearest_classes.value_counts().idxmax()
#
# print("===== K=3 without normalize =====")
# print("three nearest neighbours:", nearest_indices.tolist())
# print("three nearest neighbours classes:", nearest_classes.tolist())
# print("predicated classes:", predicted_class)
# print("Real class:", y_test_actual)
# print("is predicate true?", predicted_class == y_test_actual)
#
# ########################################
# # PART5 with normalize
# ########################################
#
# mean = x.mean()
# std = x.std()
#
# X_normalized = (x - mean) / std
#
# X_train_norm = X_normalized.iloc[:-1]
# P_test_norm = X_normalized.iloc[-1]
#
# distances_norm = np.sqrt(((X_train_norm - P_test_norm) ** 2).sum(axis=1))
# nearest_indices_norm = distances_norm.nsmallest(k).index
# nearest_classes_norm = y_train.loc[nearest_indices_norm]
# predicted_class_norm = nearest_classes_norm.value_counts().idxmax()
#
# print("===== K=3 without normalize =====")
# print("three nearest neighbours:",  nearest_indices_norm.tolist())
# print("three nearest neighbours classes:",  nearest_classes_norm.tolist())
# print("predicated classes:", predicted_class_norm)
# print("Real class:", y_test_actual)
# print("is predicate true?", predicted_class_norm == y_test_actual)

#
#
#
#
#
#

# ################################################################################
# # Question3
# ################################################################################
#
# ########################################
# # PART1
# ########################################
#
# df = pd.read_csv("iris_dataset.csv")
# df = pd.DataFrame(df)
#
# x = df[['sepal length (cm)', 'sepal width (cm)']]
# y = df['target']
#
#
# ########################################
# # PART2
# ########################################
#
# P_test = x.iloc[-1]
# y_test_actual = y.iloc[-1]
#
# X_train = x.iloc[:-1]
# y_train = y.iloc[:-1]
#
# ########################################
# # PART3
# ########################################
# #without normalize
#
# distances = np.sqrt(((X_train - P_test) ** 2).sum(axis=1))
#
#
# ########################################
# # PART4
# ########################################
#
# k = 3
# nearest_indices = distances.nsmallest(k).index
# nearest_classes = y_train.loc[nearest_indices]
#
#
# predicted_class = nearest_classes.value_counts().idxmax()
#
# print("===== K=3 without normalize =====")
# print("three nearest neighbours:", nearest_indices.tolist())
# print("three nearest neighbours classes:", nearest_classes.tolist())
# print("predicated classes:", predicted_class)
# print("Real class:", y_test_actual)
# print("is predicate true?", predicted_class == y_test_actual)
#
# ########################################
# # PART5 with normalize
# ########################################
#
# mean = x.mean()
# std = x.std()
#
# X_normalized = (x - mean) / std
#
# X_train_norm = X_normalized.iloc[:-1]
# P_test_norm = X_normalized.iloc[-1]
#
# distances_norm = np.sqrt(((X_train_norm - P_test_norm) ** 2).sum(axis=1))
# nearest_indices_norm = distances_norm.nsmallest(k).index
# nearest_classes_norm = y_train.loc[nearest_indices_norm]
# predicted_class_norm = nearest_classes_norm.value_counts().idxmax()
#
# print("===== K=3 without normalize =====")
# print("three nearest neighbours:",  nearest_indices_norm.tolist())
# print("three nearest neighbours classes:",  nearest_classes_norm.tolist())
# print("predicated classes:", predicted_class_norm)
# print("Real class:", y_test_actual)
# print("is predicate true?", predicted_class_norm == y_test_actual)






################################################################################
# Question4 - part1
################################################################################

########################################
# PART1
########################################
df = pd.read_csv("iris_dataset.csv")
########################################
# PART2
########################################
x = df[['sepal length (cm)', 'sepal width (cm)','petal length (cm)', 'petal width (cm)']]
y = df['target']

########################################
# PART3
########################################

stats = {}
for feature in x.columns:
    grouped = df.groupby('target')[feature]
    stats[feature] = {
        'min': grouped.min(),
        'max': grouped.max(),
        'mean': grouped.mean(),
        'variance': grouped.var(),
        'range': grouped.max() - grouped.min()
    }

for feature, values in stats.items():
    print(f"\nFeature: {feature}")
    for stat_name, stat_value in values.items():
        print(f"{stat_name}:\n{stat_value}\n")

########################################
# PART4
########################################
def normal_pdf(x, mean, var):
    return (1 / np.sqrt(2 * np.pi * var)) * np.exp(- (x - mean)**2 / (2 * var))

########################################
# PART5
########################################
X_test = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
y_test = df['target']

predictions = []

for i, row in X_test.iterrows():
    class_probs = {}
    for cls in df['target'].unique():
        prob = 1
        for feature in x.columns:
            mean = stats[feature]['mean'][cls]
            var = stats[feature]['variance'][cls]
            prob *= normal_pdf(row[feature], mean, var)
        class_probs[cls] = prob
    predicted_class = max(class_probs, key=class_probs.get)
    predictions.append(predicted_class)

# Accuracy
accuracy = (sum(predictions == y_test) / len(y_test)) * 100
print(f"\nModel Accuracy: {accuracy:.2f}%")



################################################################################
# Question4 - part2
################################################################################

########################################
# PART1
########################################

def normal_pdf(x_val, mean, var):
    return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-(x_val - mean)**2 / (2 * var))

df_test = pd.read_csv("iris_test_samples.csv")

df_test.columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)','label']

X_test = df_test[['petal length (cm)']]
y_test_actual = df_test['label']
print(df_test.head())

predictions = []
for i, row in X_test.iterrows():
    class_probs = {}
    for cls in y.unique():
        mean = stats['sepal length (cm)']['mean'][cls]
        var = stats['sepal length (cm)']['variance'][cls]
        class_probs[cls] = normal_pdf(row['petal length (cm)'], mean, var)
    predicted_class = max(class_probs, key=class_probs.get)
    predictions.append(predicted_class)

accuracy = (sum(np.array(predictions) == np.array(y_test_actual)) / len(y_test_actual)) * 100
print(f"\nAccuracy using petal_length only: {accuracy:.2f}%")


########################################
# PART2
########################################

X_test = df_test[['petal length (cm)', 'petal width (cm)']]
predictions = []
for i, row in X_test.iterrows():
    class_probs = {}
    for cls in y.unique():
        prob1 = normal_pdf(row['petal length (cm)'], stats['sepal length (cm)']['mean'][cls], stats['sepal length (cm)']['variance'][cls])
        prob2 = normal_pdf(row['petal width (cm)'], stats['sepal width (cm)']['mean'][cls], stats['sepal width (cm)']['variance'][cls])
        class_probs[cls] = (prob1 + prob2) / 2
    predicted_class = max(class_probs, key=class_probs.get)
    predictions.append(predicted_class)

accuracy = (sum(np.array(predictions) == np.array(y_test_actual)) / len(y_test_actual)) * 100
print(f"\nAccuracy using petal_length + petal_width: {accuracy:.2f}%")


########################################
# PART3
########################################


X_test = df_test[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
predictions = []
for i, row in X_test.iterrows():
    class_probs = {}
    for cls in y.unique():
        probs = []
        for feature in X_test.columns:
            probs.append(normal_pdf(row[feature], stats[feature]['mean'][cls], stats[feature]['variance'][cls]))
        top2_mean = np.mean(sorted(probs, reverse=True)[:2])
        class_probs[cls] = top2_mean
    predicted_class = max(class_probs, key=class_probs.get)
    predictions.append(predicted_class)

accuracy = (sum(np.array(predictions) == np.array(y_test_actual)) / len(y_test_actual)) * 100
print(f"\nAccuracy using top-2 probabilities among 4 features: {accuracy:.2f}%")