import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def classification_models():

    print("\n" + "=" * 55)
    print("                  CLASSIFICATION")
    print("=" * 55)

    # Study hours and attendance
    X = np.array([
        [1, 50],
        [2, 55],
        [2, 60],
        [3, 62],
        [3, 65],
        [4, 68],
        [4, 70],
        [5, 72],
        [5, 75],
        [6, 78],
        [6, 80],
        [7, 82],
        [7, 85],
        [8, 88],
        [8, 90],
        [9, 92],
        [9, 95],
        [10, 97]
    ])

    # 0 = Fail, 1 = Pass
    y = np.array([
        0, 0, 0, 0, 0,
        0, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1
    ])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # ==================================================
    # LOGISTIC REGRESSION
    # ==================================================

    logistic_model = LogisticRegression()

    logistic_model.fit(X_train, y_train)

    logistic_prediction = logistic_model.predict(X_test)

    logistic_accuracy = accuracy_score(
        y_test,
        logistic_prediction
    )

    print("\n--- Logistic Regression ---")
    print(
        "Accuracy:",
        round(logistic_accuracy * 100, 2),
        "%"
    )

    # ==================================================
    # KNN
    # ==================================================

    knn_model = KNeighborsClassifier(n_neighbors=3)

    knn_model.fit(X_train, y_train)

    knn_prediction = knn_model.predict(X_test)

    knn_accuracy = accuracy_score(
        y_test,
        knn_prediction
    )

    print("\n--- K-Nearest Neighbors (KNN) ---")
    print(
        "Accuracy:",
        round(knn_accuracy * 100, 2),
        "%"
    )

    # ==================================================
    # SAMPLE PREDICTION
    # ==================================================

    student = np.array([[6, 80]])

    logistic_result = logistic_model.predict(student)[0]
    knn_result = knn_model.predict(student)[0]

    print("\n--- Sample Student Prediction ---")
    print("Study Hours:", student[0][0])
    print("Attendance:", student[0][1])

    print(
        "Logistic Regression:",
        "PASS" if logistic_result == 1 else "FAIL"
    )

    print(
        "KNN:",
        "PASS" if knn_result == 1 else "FAIL"
    )

    # ==================================================
    # GRAPH
    # ==================================================

    plt.figure(figsize=(8, 5))

    plt.scatter(
        X[y == 0, 0],
        X[y == 0, 1],
        label="Fail"
    )

    plt.scatter(
        X[y == 1, 0],
        X[y == 1, 1],
        label="Pass"
    )

    plt.xlabel("Study Hours")
    plt.ylabel("Attendance (%)")
    plt.title("Student Classification")

    plt.legend()
    plt.grid()

    plt.show()