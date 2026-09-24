import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from knee_for_DBSCAN import find_eps

# Load Dataset
data = pd.read_csv(r"preprocessed_placement.csv")

# Take maximum 1000 samples
if len(data) > 1000:
    data = data.sample(n=1000, random_state=42)

# Remove target column
X = data.drop("PlacementStatus", axis=1)

# Create DBSCAN model
model = DBSCAN(eps=find_eps(X, min_samples=5), min_samples=5)

# Fit DBSCAN
labels = model.fit_predict(X)

# Find Core Points
core = model.core_sample_indices_

# Find Noise Points
noise = labels == -1

# Find Border Points
border = ~noise & ~pd.Series(range(len(X))).isin(core)

# Print Results
print("Clusters:", len(set(labels)) - (1 if -1 in labels else 0))
print("Core:", len(core))
print("Border:", border.sum())
print("Noise:", noise.sum())

# Plot DBSCAN Clusters
plt.scatter(
    X.iloc[:, 0],
    X.iloc[:, 1],
    c=labels
)

plt.xlabel(X.columns[0])
plt.ylabel(X.columns[1])
plt.title("DBSCAN Clustering")

plt.show()