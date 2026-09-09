import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score


def clustering_demo():

    print("\n" + "=" * 60)
    print("                  CLUSTERING")
    print("=" * 60)

    # --------------------------------------------------
    # DATASET
    # --------------------------------------------------

    # Features:
    # Study Hours and Attendance

    data = np.array([
        [2, 55],
        [2.5, 58],
        [3, 60],
        [3.2, 62],
        [3.5, 65],

        [6, 75],
        [6.5, 78],
        [7, 80],
        [7.2, 82],
        [7.5, 85],

        [9, 90],
        [9.5, 92],
        [10, 94],
        [10.5, 96],
        [11, 98],

        [5, 70],
        [8, 88],
        [4, 64]
    ])

    print("\nFeatures:")
    print("Study Hours and Attendance")

    # --------------------------------------------------
    # FEATURE SCALING
    # --------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(data)

    print("\nFeature scaling completed.")

    # ==================================================
    # K-MEANS CLUSTERING
    # ==================================================

    print("\n--- K-Means Clustering ---")

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    kmeans_labels = kmeans.fit_predict(X_scaled)

    kmeans_score = silhouette_score(
        X_scaled,
        kmeans_labels
    )

    print("Number of Clusters:", 3)
    print("Cluster Labels:", kmeans_labels)

    print(
        "Silhouette Score:",
        round(kmeans_score, 3)
    )

    # ==================================================
    # HIERARCHICAL CLUSTERING
    # ==================================================

    print("\n--- Hierarchical Clustering ---")

    hierarchical = AgglomerativeClustering(
        n_clusters=3
    )

    hierarchical_labels = hierarchical.fit_predict(
        X_scaled
    )

    hierarchical_score = silhouette_score(
        X_scaled,
        hierarchical_labels
    )

    print("Number of Clusters:", 3)
    print("Cluster Labels:", hierarchical_labels)

    print(
        "Silhouette Score:",
        round(hierarchical_score, 3)
    )

    # ==================================================
    # DBSCAN CLUSTERING
    # ==================================================

    print("\n--- DBSCAN Clustering ---")

    dbscan = DBSCAN(
        eps=0.55,
        min_samples=2
    )

    dbscan_labels = dbscan.fit_predict(
        X_scaled
    )

    print("Cluster Labels:", dbscan_labels)

    # Count clusters excluding noise
    unique_labels = set(dbscan_labels)

    cluster_count = len(
        unique_labels - {-1}
    )

    noise_count = list(
        dbscan_labels
    ).count(-1)

    print(
        "Number of Clusters:",
        cluster_count
    )

    print(
        "Number of Noise Points:",
        noise_count
    )

    # Calculate silhouette score
    # only when at least 2 clusters exist

    if cluster_count >= 2:

        dbscan_score = silhouette_score(
            X_scaled,
            dbscan_labels
        )

        print(
            "Silhouette Score:",
            round(dbscan_score, 3)
        )

    else:

        print(
            "Silhouette Score: Not available"
        )

    # ==================================================
    # CLUSTERING INTERPRETATION
    # ==================================================

    print("\n--- Clustering Interpretation ---")

    print(
        "K-Means: Groups data into a fixed number of clusters."
    )

    print(
        "Hierarchical: Groups similar data points into a hierarchy of clusters."
    )

    print(
        "DBSCAN: Groups dense data points and identifies noise or outliers."
    )

    # ==================================================
    # K-MEANS GRAPH
    # ==================================================

    plt.figure(figsize=(8, 5))

    plt.scatter(
        data[:, 0],
        data[:, 1],
        c=kmeans_labels
    )

    plt.xlabel("Study Hours")
    plt.ylabel("Attendance (%)")
    plt.title("K-Means Clustering")

    plt.grid()
    plt.show()

    # ==================================================
    # HIERARCHICAL GRAPH
    # ==================================================

    plt.figure(figsize=(8, 5))

    plt.scatter(
        data[:, 0],
        data[:, 1],
        c=hierarchical_labels
    )

    plt.xlabel("Study Hours")
    plt.ylabel("Attendance (%)")
    plt.title("Hierarchical Clustering")

    plt.grid()
    plt.show()

    # ==================================================
    # DBSCAN GRAPH
    # ==================================================

    plt.figure(figsize=(8, 5))

    plt.scatter(
        data[:, 0],
        data[:, 1],
        c=dbscan_labels
    )

    plt.xlabel("Study Hours")
    plt.ylabel("Attendance (%)")
    plt.title("DBSCAN Clustering")

    plt.grid()
    plt.show()

    print("\nClustering completed successfully.")