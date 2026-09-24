import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
try:
    from kneed import KneeLocator
    HAS_KNEED = True
except ImportError:
    HAS_KNEED = False

# Base directory for charts
CHARTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

def load_clean_dataset():
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preprocessed_placement.csv")
    if not os.path.exists(filepath):
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "placement_predict_50k Dataset (3) 1(in).csv")
    
    df = pd.read_csv(filepath)
    numeric_df = df.select_dtypes(include=[np.number]).fillna(0)
    
    # Extract student placement features
    X_df = pd.DataFrame()
    
    if "CGPA" in numeric_df.columns:
        X_df["CGPA"] = numeric_df["CGPA"]
    elif "SGPA_Sem1" in numeric_df.columns:
        X_df["CGPA"] = numeric_df[["SGPA_Sem1", "SGPA_Sem2", "SGPA_Sem3"]].mean(axis=1)
    else:
        X_df["CGPA"] = np.random.uniform(6.0, 9.5, size=len(numeric_df))
        
    if "CodingTestScore" in numeric_df.columns:
        X_df["Coding Test Score"] = numeric_df["CodingTestScore"]
    else:
        X_df["Coding Test Score"] = np.random.uniform(50, 95, size=len(numeric_df))

    if "AptitudeTestScore" in numeric_df.columns:
        X_df["Aptitude Test Score"] = numeric_df["AptitudeTestScore"]
    else:
        X_df["Aptitude Test Score"] = np.random.uniform(55, 98, size=len(numeric_df))

    if "Internships" in numeric_df.columns:
        X_df["Internships"] = numeric_df["Internships"]
    else:
        X_df["Internships"] = np.random.randint(0, 4, size=len(numeric_df))

    return df, X_df

# =========================================================
# 1. K-MEANS CLUSTERING
# =========================================================
def run_kmeans_pipeline(method="Manual", k=3):
    df_raw, X_df = load_clean_dataset()
    
    if len(X_df) > 5000:
        X_sample = X_df.sample(n=5000, random_state=42).reset_index(drop=True)
    else:
        X_sample = X_df.copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sample[["CGPA", "Coding Test Score"]])

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    wcss = round(kmeans.inertia_, 2)
    sil_score = round(silhouette_score(X_scaled, labels), 4) if len(np.unique(labels)) > 1 else "N/A"
    
    cluster_counts = pd.Series(labels).value_counts().to_dict()
    distribution = [
        {"cluster": i, "count": cluster_counts.get(i, 0)}
        for i in range(k)
    ]

    # Generate K-Means Scatter Chart for Student Placement
    plt.figure(figsize=(10, 6), dpi=120)
    scatter = plt.scatter(
        X_sample["CGPA"],
        X_sample["Coding Test Score"],
        c=labels,
        cmap="viridis",
        alpha=0.75,
        edgecolor='none',
        s=30
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label("Cluster Group", fontsize=11)
    plt.title(f"K-Means Clusters: Coding Test Score vs CGPA (K = {k})", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Cumulative Grade Point Average (CGPA)", fontsize=11)
    plt.ylabel("Coding Test Score (out of 100)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    
    chart_path = os.path.join(CHARTS_DIR, "kmeans_scatter.png")
    plt.savefig(chart_path)
    plt.close()

    return {
        "method": method,
        "selected_k": k,
        "wcss": f"{wcss:,.2f}",
        "silhouette_score": sil_score,
        "total_records": f"{len(X_sample):,}",
        "distribution": distribution,
        "chart_url": "/static/charts/kmeans_scatter.png"
    }

# =========================================================
# 2. HIERARCHICAL CLUSTERING
# =========================================================
def run_hierarchical_pipeline(method="ward", k=2):
    df_raw, X_df = load_clean_dataset()
    
    X_sample = X_df.sample(n=100, random_state=42).reset_index(drop=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sample[["CGPA", "Coding Test Score"]])

    Z = linkage(X_scaled, method=method)
    labels = fcluster(Z, t=k, criterion="maxclust")
    
    sil_score = round(silhouette_score(X_scaled, labels), 4) if len(np.unique(labels)) > 1 else "N/A"
    cut_height = round(Z[-k+1, 2], 2) if k > 1 and len(Z) >= k else 88.67

    # 1. Dendrogram Plot
    fig, ax = plt.subplots(figsize=(11, 6), dpi=120)
    ddata = dendrogram(Z, ax=ax, truncate_mode=None, color_threshold=Z[-k+1, 2] if k>1 else 0)
    
    ax.axhline(y=cut_height, color="r", linestyle="--", linewidth=1.8, label=f"Cut Height = {cut_height} (K={k})")
    
    for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
        x = 0.5 * sum(i[1:3])
        y = d[1]
        if y > cut_height * 0.15:
            ax.annotate(f"h={y:.1f}", (x, y), xytext=(0, 3), textcoords="offset points",
                        va="bottom", ha="center", fontsize=8, fontweight="bold",
                        bbox=dict(boxstyle="square,pad=0.2", fc="aliceblue", ec="navy", lw=0.6))

    ax.set_title("Hierarchical Clustering Dendrogram for Student Placement Profiles", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Student Sample Index / Cluster Group", fontsize=11)
    ax.set_ylabel("Distance (Dissimilarity / Merge Height)", fontsize=11)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()
    
    dendro_path = os.path.join(CHARTS_DIR, "hierarchical_dendrogram.png")
    plt.savefig(dendro_path)
    plt.close()

    # 2. Clusters Scatter Plot
    plt.figure(figsize=(10, 6), dpi=120)
    X_scatter = X_df.sample(n=1000, random_state=42).reset_index(drop=True)
    X_scatter_scaled = scaler.transform(X_scatter[["CGPA", "Coding Test Score"]])
    scatter_labels = fcluster(linkage(X_scatter_scaled, method=method), t=k, criterion="maxclust")

    colors = ["#3182bd", "#e6550d", "#31a354", "#756bb1", "#636363"]
    for cl in range(1, k + 1):
        mask = scatter_labels == cl
        plt.scatter(
            X_scatter.loc[mask, "CGPA"],
            X_scatter.loc[mask, "Coding Test Score"],
            c=colors[(cl - 1) % len(colors)],
            label=f"Cluster {cl}",
            alpha=0.75,
            edgecolors='none',
            s=35
        )

    plt.title("Hierarchical Clusters: Coding Test Score vs CGPA", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Cumulative Grade Point Average (CGPA)", fontsize=11)
    plt.ylabel("Coding Test Score (out of 100)", fontsize=11)
    plt.legend(title="Student Clusters", loc="upper left", frameon=True)
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()

    scatter_path = os.path.join(CHARTS_DIR, "hierarchical_scatter.png")
    plt.savefig(scatter_path)
    plt.close()

    return {
        "selected_k": k,
        "linkage_method": method.capitalize(),
        "cut_height": cut_height,
        "silhouette_score": sil_score,
        "dendrogram_url": "/static/charts/hierarchical_dendrogram.png",
        "scatter_url": "/static/charts/hierarchical_scatter.png"
    }

# =========================================================
# 3. DBSCAN CLUSTERING
# =========================================================
def run_dbscan_pipeline(min_samples=5):
    df_raw, X_df = load_clean_dataset()
    
    X_sample = X_df.sample(n=1000, random_state=42).reset_index(drop=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sample[["CGPA", "Coding Test Score"]])

    neighbors = NearestNeighbors(n_neighbors=min_samples, algorithm='ball_tree')
    neighbors.fit(X_scaled)
    distances, _ = neighbors.kneighbors(X_scaled)
    k_distances = np.sort(distances[:, -1])
    x_indices = np.arange(len(k_distances))

    if HAS_KNEED:
        kneedle = KneeLocator(x_indices, k_distances, curve="convex", direction="increasing")
        knee_idx = kneedle.knee if kneedle.knee is not None else int(len(k_distances) * 0.95)
        eps = round(k_distances[knee_idx], 4)
    else:
        knee_idx = int(len(k_distances) * 0.95)
        eps = round(k_distances[knee_idx], 4)

    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_scaled)
    
    core_indices = set(db.core_sample_indices_)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    n_core = len(core_indices)
    n_border = len(X_sample) - n_core - n_noise

    sil_score = round(silhouette_score(X_scaled, labels), 4) if n_clusters > 1 else "N/A"

    # Knee Chart
    fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
    ax.plot(x_indices, k_distances, color="#00a896", linewidth=2, label="k-Distance")
    ax.axvline(x=knee_idx, color="crimson", linestyle="--", linewidth=1.5, label=f"Knee Point (Index {knee_idx})")
    ax.axhline(y=eps, color="dodgerblue", linestyle=":", linewidth=1.5, label=f"Optimal Eps = {eps}")
    ax.set_title(f"K-Distance Graph (DBSCAN Epsilon Detection - Eps: {eps})", fontsize=11, fontweight="bold")
    ax.set_xlabel("Student Records (Sorted by Distance)", fontsize=10)
    ax.set_ylabel(f"{min_samples}th Nearest Neighbors Distance", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()
    
    knee_chart_path = os.path.join(CHARTS_DIR, "dbscan_knee.png")
    plt.savefig(knee_chart_path)
    plt.close()

    # DBSCAN Scatter Plot
    fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
    
    noise_mask = (labels == -1)
    ax.scatter(
        X_sample.loc[noise_mask, "CGPA"],
        X_sample.loc[noise_mask, "Coding Test Score"],
        c="#a0a0a0",
        marker="x",
        s=30,
        alpha=0.6,
        label="Noise / Outliers (-1)"
    )
    
    unique_labels = set(labels) - {-1}
    colors = ["#483d8b", "#20b2aa", "#9370db", "#3cb371"]
    for i, l in enumerate(unique_labels):
        cluster_mask = (labels == l)
        ax.scatter(
            X_sample.loc[cluster_mask, "CGPA"],
            X_sample.loc[cluster_mask, "Coding Test Score"],
            c=colors[i % len(colors)],
            marker="o",
            s=35,
            alpha=0.8,
            label=f"Cluster {l + 1} (Core)"
        )

    ax.set_title("DBSCAN Clustering: Coding Test Score vs CGPA", fontsize=11, fontweight="bold")
    ax.set_xlabel("CGPA", fontsize=10)
    ax.set_ylabel("Coding Test Score", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()

    dbscan_scatter_path = os.path.join(CHARTS_DIR, "dbscan_scatter.png")
    plt.savefig(dbscan_scatter_path)
    plt.close()

    # Summary Table Data for Placement Prediction
    summary_table = []
    
    if n_noise > 0:
        noise_data = X_sample[noise_mask]
        summary_table.append({
            "group": "Noise Points (-1)",
            "is_noise": True,
            "total_points": n_noise,
            "core_points": 0,
            "border_points": 0,
            "avg_cgpa": f"{noise_data['CGPA'].mean():.2f}",
            "avg_coding": f"{noise_data['Coding Test Score'].mean():.1f}",
            "avg_aptitude": f"{noise_data['Aptitude Test Score'].mean():.1f}",
            "avg_internships": f"{noise_data['Internships'].mean():.1f}"
        })

    for l in sorted(list(unique_labels)):
        c_mask = (labels == l)
        c_data = X_sample[c_mask]
        c_core = sum(1 for idx in c_data.index if idx in core_indices)
        c_border = len(c_data) - c_core
        summary_table.append({
            "group": f"Cluster {l + 1}",
            "is_noise": False,
            "total_points": len(c_data),
            "core_points": c_core,
            "border_points": c_border,
            "avg_cgpa": f"{c_data['CGPA'].mean():.2f}",
            "avg_coding": f"{c_data['Coding Test Score'].mean():.1f}",
            "avg_aptitude": f"{c_data['Aptitude Test Score'].mean():.1f}",
            "avg_internships": f"{c_data['Internships'].mean():.1f}"
        })

    return {
        "clusters_count": n_clusters,
        "core_points": n_core,
        "border_points": n_border,
        "noise_points": n_noise,
        "epsilon": eps,
        "silhouette_score": sil_score,
        "knee_chart_url": "/static/charts/dbscan_knee.png",
        "scatter_chart_url": "/static/charts/dbscan_scatter.png",
        "summary_table": summary_table
    }
