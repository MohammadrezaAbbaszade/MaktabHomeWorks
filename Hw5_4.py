import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


########################################
# PART1
########################################

df = pd.read_csv("iris_dataset.csv")
df = pd.DataFrame(df)

########################################
# PART2
########################################
df.columns = ["sepal_length", "sepal_width",
              "petal_length", "petal_width", "species"]

x = df[['sepal_length', 'sepal_width']]
y = df['species']


########################################
# PART3
########################################

stats = {}

for feature in x.columns:
    stats[feature] = {
        'min': df.groupby('Species')[feature].min(),
        'max': df.groupby('Species')[feature].max(),
        'mean': df.groupby('Species')[feature].mean(),
        'variance': df.groupby('Species')[feature].var(),
        'range': df.groupby('Species')[feature].max() - df.groupby('Species')[feature].min()
    }


#features printing
for feature, values in stats.items():
    print(f"feature: {feature}")
    for stat_name, stat_value in values.items():
        print(f"  {stat_name}: \n{stat_value}\n")

########################################
# PART4
########################################
def normal_pdf(x, mean, var):
    return (1 / np.sqrt(2 * np.pi * var)) * np.exp(- (x - mean)**2 / (2 * var))


probabilities = {}

for feature in x.columns:
    probabilities[feature] = {}
    for species in df['Species'].unique():
        mean = stats[feature]['mean'][species]
        var = stats[feature]['variance'][species]


#Result
for feature, species_probs in probabilities.items():
    print(f"\nfeature: {feature}")
    for species, prob in species_probs.items():
        print(f"  Member in class {species}: \n{prob[:5]} ...")


########################################
# PART5
########################################
X_test = df[['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']]
y_test = df['Species']

predictions = []

for i, row in X_test.iterrows():
    class_probs = {}
    for species in df['Species'].unique():
        total_prob = 1
        for feature in x.columns:
            mean = stats[feature]['mean'][species]
            var = stats[feature]['variance'][species]
            total_prob *= normal_pdf(row[feature], mean, var)
        class_probs[species] = total_prob

 #computing max class with high priority of probabilities
    predicted_class = max(class_probs, key=class_probs.get)
    predictions.append(predicted_class)

    # Accuracy calculation
    correct_answers = sum([pred == actual for pred, actual in zip(predictions, y_test)])
    accuracy = (correct_answers / len(y_test)) * 100

    print(f"\nModel Accuracy: {accuracy:.2f}%")
