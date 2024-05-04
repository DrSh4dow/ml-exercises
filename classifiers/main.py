from pandas import DataFrame
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np

iris = datasets.load_iris()
X: DataFrame = iris.data[:, [2, 3]]  # type: ignore

y: DataFrame = iris.target  # type: ignore


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)

print("Labels counts in y:", np.bincount(y))
print("Labels counts in y_train:", np.bincount(y_train))
print("Labels counts in y_test:", np.bincount(y_test))


sc = StandardScaler()

sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

ppn = Perceptron(eta0=0.1, random_state=1)

ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)

print(accuracy_score(y_test, y_pred))
