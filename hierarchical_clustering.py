import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

data = pd.read_csv("placement_predict_50k Dataset.csv")

X = data.drop("PlacementStatus", axis=1)

# Convert categorical columns like Male/Female into numbers
X = pd.get_dummies(X, drop_first=True)

# Remove NaN and infinite values
X = X.replace([float("inf"), float("-inf")], float("nan"))
X = X.dropna()

# Take 100 students
X = X.sample(100, random_state=42)

methods = ["single", "complete", "average", "ward"]

for method in methods:

    if method == "ward":
        Z = linkage(X, method="ward")
    else:
        Z = linkage(X, method=method, metric="euclidean")

    # Print distances in PyCharm terminal
    print("\n" + "=" * 60)
    print(method.upper(), "LINKAGE")
    print("=" * 60)

    for i, row in enumerate(Z, start=1):
        print(
            "Merge:", i,
            "| Cluster 1:", int(row[0]),
            "| Cluster 2:", int(row[1]),
            "| Distance:", round(row[2], 4)
        )

    # Dendrogram
    plt.figure(figsize=(12, 6))

    dendrogram(Z)

    plt.title(method.capitalize() + " Linkage")
    plt.xlabel("Students")
    plt.ylabel("Distance")

    plt.tight_layout()
    plt.show()