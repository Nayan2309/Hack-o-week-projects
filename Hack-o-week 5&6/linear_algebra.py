import numpy as np


def linear_algebra():
    print("\n" + "=" * 50)
    print("              LINEAR ALGEBRA")
    print("=" * 50)

    # ---------------- VECTORS ----------------
    print("\n--- Vectors ---")

    vector_a = np.array([2, 3])
    vector_b = np.array([4, 1])

    print("Vector A:", vector_a)
    print("Vector B:", vector_b)
    print("A + B:", vector_a + vector_b)
    print("A - B:", vector_a - vector_b)

    magnitude = np.linalg.norm(vector_a)
    print("Magnitude of A:", round(magnitude, 2))

    # ---------------- DOT PRODUCT ----------------
    print("\n--- Dot Product ---")

    dot_product = np.dot(vector_a, vector_b)

    print("A · B =", dot_product)

    # ---------------- MATRICES ----------------
    print("\n--- Matrices ---")

    matrix_a = np.array([
        [1, 2],
        [3, 4]
    ])

    matrix_b = np.array([
        [5, 6],
        [7, 8]
    ])

    print("Matrix A:")
    print(matrix_a)

    print("\nMatrix B:")
    print(matrix_b)

    print("\nA + B:")
    print(matrix_a + matrix_b)

    print("\nA × B:")
    print(np.dot(matrix_a, matrix_b))

    # ---------------- EIGENVALUES ----------------
    print("\n--- Eigenvalues ---")

    eigenvalues, eigenvectors = np.linalg.eig(matrix_a)

    print("Eigenvalues:", np.round(eigenvalues, 2))

    print("Eigenvectors:")
    print(np.round(eigenvectors, 2))