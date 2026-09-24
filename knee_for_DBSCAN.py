from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
import pandas as pd
from matplotlib import pyplot as plt


def find_eps(X, min_samples=5):
    data = pd.read_csv(r"preprocessed_placement.csv")

    # Take maximum 1000 samples
    if len(data) > 1000:
        data = data.sample(n=1000, random_state=42)

    # Remove target column
    X = data.drop("PlacementStatus", axis=1)

    min_samples = 5

    neighbors = NearestNeighbors(n_neighbors=min_samples, algorithm='ball_tree')
    neighbors.fit(X)

    distances, indices = neighbors.kneighbors(X)

    k_distances = sorted(distances[:, -1])

    plt.figure(figsize=(15, 10))
    plt.plot(k_distances)
    plt.xlabel("Data Points")
    plt.ylabel("5th Nearest Neighbors Distance")
    plt.title("K-Distance Graph")
    plt.grid()
    plt.show()

    x = range(len(k_distances))

    knee = KneeLocator(
        x,
        k_distances,
        curve="convex",
        direction="increasing"
    )

    eps = round(k_distances[knee.knee], 4)

    return eps

