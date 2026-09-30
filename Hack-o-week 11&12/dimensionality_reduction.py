import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def dimensionality_reduction():

    print("\n" + "=" * 60)
    print("          DIMENSIONALITY REDUCTION")
    print("=" * 60)

    # --------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------

    iris = load_iris()

    X = iris.data
    y = iris.target

    print("\nDataset: Iris")
    print("Original number of features:", X.shape[1])
    print("Number of samples:", X.shape[0])

    print("\nOriginal Features:")
    print(iris.feature_names)

    # --------------------------------------------------
    # FEATURE SCALING
    # --------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    print("\nFeature scaling completed.")

    # ==================================================
    # PCA
    # ==================================================

    print("\n--- PCA (Principal Component Analysis) ---")

    pca = PCA(n_components=2)

    X_pca = pca.fit_transform(X_scaled)

    print("Original dimensions:", X.shape[1])
    print("Reduced dimensions:", X_pca.shape[1])

    print(
        "Explained Variance Ratio:",
        np.round(pca.explained_variance_ratio_, 3)
    )

    print(
        "Total Variance Explained:",
        round(
            pca.explained_variance_ratio_.sum() * 100,
            2
        ),
        "%"
    )

    # --------------------------------------------------
    # PCA VISUALIZATION
    # --------------------------------------------------

    plt.figure(figsize=(8, 5))

    for class_value in np.unique(y):

        plt.scatter(
            X_pca[y == class_value, 0],
            X_pca[y == class_value, 1],
            label=iris.target_names[class_value]
        )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA - Iris Dataset")
    plt.legend()
    plt.grid()

    plt.show()

    # ==================================================
    # t-SNE
    # ==================================================

    print("\n--- t-SNE ---")

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=42,
        max_iter=1000
    )

    X_tsne = tsne.fit_transform(X_scaled)

    print("Original dimensions:", X.shape[1])
    print("Reduced dimensions:", X_tsne.shape[1])

    print("t-SNE transformation completed.")

    # --------------------------------------------------
    # t-SNE VISUALIZATION
    # --------------------------------------------------

    plt.figure(figsize=(8, 5))

    for class_value in np.unique(y):

        plt.scatter(
            X_tsne[y == class_value, 0],
            X_tsne[y == class_value, 1],
            label=iris.target_names[class_value]
        )

    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.title("t-SNE - Iris Dataset")
    plt.legend()
    plt.grid()

    plt.show()

    # ==================================================
    # INTERPRETATION
    # ==================================================

    print("\n--- Interpretation ---")

    print(
        "PCA reduces dimensions while preserving maximum variance."
    )

    print(
        "PCA is useful for data compression, visualization and feature reduction."
    )

    print(
        "t-SNE converts high-dimensional data into a 2D or 3D representation."
    )

    print(
        "t-SNE is mainly useful for visualizing similar data points and clusters."
    )

    print(
        "PCA is a linear dimensionality reduction technique."
    )

    print(
        "t-SNE is mainly a visualization technique that captures local relationships."
    )

    print("\nDimensionality reduction completed successfully.")