import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from load_data import load_data


def run_hierarchical_clustering():
    # 1. Load Data
    data = load_data()

    # 2. Exclude Non-Feature and Target Columns
    excluded_columns = [
        "StudentID",
        "PlacementStatus",
        "IsAnomaly",
        "Salary Package",
        "CGPA_Tier",
    ]

    columns_to_drop = [col for col in excluded_columns if col in data.columns]
    X = data.drop(columns=columns_to_drop)

    # 3. Handle Missing Values
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    categorical_cols = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns

    for col in numeric_cols:
        if X[col].isnull().sum() > 0:
            X[col] = X[col].fillna(X[col].median())

    for col in categorical_cols:
        if X[col].isnull().sum() > 0:
            mode_val = X[col].mode()
            X[col] = X[col].fillna(mode_val.iloc[0] if not mode_val.empty else "Unknown")

    # 4. Convert categorical columns to dummy numeric variables
    X = pd.get_dummies(X, drop_first=True)

    # Clean any leftover NaN / Inf values
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # 5. Take a sample of 100 students for clean dendrogram visualization
    X_sample = X.sample(n=100, random_state=42)

    # 6. Scale numerical features so Euclidean distance is valid across all attributes
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sample)

    # Convert back to DataFrame to maintain structure
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_sample.columns, index=X_sample.index)

    methods = ["single", "complete", "average", "ward"]

    # Output folder for saving charts
    charts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "charts")
    os.makedirs(charts_dir, exist_ok=True)

    results = {}

    for method in methods:
        if method == "ward":
            Z = linkage(X_scaled_df, method="ward")
        else:
            Z = linkage(X_scaled_df, method=method, metric="euclidean")

        print("\n" + "=" * 60)
        print(f"{method.upper()} LINKAGE")
        print("=" * 60)

        for i, row in enumerate(Z[:10], start=1):  # Print first 10 merges
            print(
                f"Merge {i:2d} | Cluster 1: {int(row[0]):3d} | Cluster 2: {int(row[1]):3d} | Distance: {row[2]:.4f}"
            )
        print(f"... total merges: {len(Z)}")

        # Create Dendrogram
        plt.figure(figsize=(12, 6))
        dendrogram(Z)
        plt.title(f"{method.capitalize()} Linkage Dendrogram")
        plt.xlabel("Student Index")
        plt.ylabel("Distance")
        plt.tight_layout()

        # Save plot image
        save_path = os.path.join(charts_dir, f"dendrogram_{method}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved plot: {save_path}")

        results[method] = Z

    return results


if __name__ == "__main__":
    run_hierarchical_clustering()