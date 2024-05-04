# load iris dataset
import pandas as pd
import numpy as np


from plot_decision_regions import plot_decision_regions
from perceptron_class import Perceptron


dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

print("From URL:", dataset_url)

print("Loading dataset...")
df = pd.read_csv(dataset_url, header=None, encoding="utf-8")
print("Dataset loaded successfully.")

print(df.head(10))

target_values = df.iloc[0:100, 4].values

target_values = np.where(target_values == "Iris-setosa", 0, 1)

feature_matrix = df.iloc[0:100, [0, 2]].values


# perceptron
ppn = Perceptron(eta=0.1, n_iter=10)

ppn.fit(feature_matrix, target_values)

plot_decision_regions(feature_matrix, target_values, ppn)
