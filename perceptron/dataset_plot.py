import matplotlib.pyplot as plt
from pandas import DataFrame


def dataset_plot(feature_matrix: DataFrame):
    """Plot the dataset."""

    # Plot data
    plt.scatter(
        feature_matrix[0:50, 0],
        feature_matrix[0:50, 1],
        color="red",
        marker="o",
        label="Setosa",
    )

    plt.scatter(
        feature_matrix[50:100, 0],
        feature_matrix[50:100, 1],
        color="blue",
        marker="s",
        label="Versicolor",
    )

    plt.xlabel("sepal length [cm]")
    plt.ylabel("petal length [cm]")

    plt.legend(loc="upper left")

    plt.show()
