from typing import List
import numpy as np
from numpy._typing import ArrayLike


class Perceptron:
    """Perceptron classifier."""

    eta: float
    """Learning rate"""

    n_iter: int
    """Passes over the training dataset."""

    random_state: int
    """Random number generator seed for random weight initialization."""

    w_: np.ndarray  # 1d-array
    """Weights after fitting."""
    b_: np.double  # Scalar
    """Bias unit after fitting."""
    errors_: List[int]  # list
    """Number of misclassifications (updates) in each epoch."""

    def __init__(self, eta=0.01, n_iter=50, random_state=1) -> None:
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit training data"""
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.double(0.0)
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for x_i, target in zip(X, y):
                update: np.double = self.eta * (target - self.predict(x_i))
                self.w_ += update * x_i
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X: ArrayLike):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_

    def predict(self, X: ArrayLike):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)
