import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    accuracy_score
)


def model_evaluation():

    print("\n" + "=" * 60)
    print("                  MODEL EVALUATION")
    print("=" * 60)

    # ==================================================
    # DATASET
    # ==================================================

    # Features:
    # Study Hours, Attendance
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
        [10, 97],
        [4, 66],
        [6, 81]
    ])

    # 0 = Fail
    # 1 = Pass
    y = np.array([
        0, 0, 0, 0, 0,
        0, 0, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 0, 1
    ])

    # ==================================================
    # TRAIN / TEST SPLIT
    # ==================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    print("\n--- Train/Test Split ---")
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # ==================================================
    # MODEL
    # ==================================================

    model = LogisticRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    # ==================================================
    # ACCURACY
    # ==================================================

    accuracy = accuracy_score(y_test, y_pred)

    print("\n--- Accuracy ---")
    print("Accuracy:", round(accuracy * 100, 2), "%")

    # ==================================================
    # CONFUSION MATRIX
    # ==================================================

    cm = confusion_matrix(y_test, y_pred)

    print("\n--- Confusion Matrix ---")
    print(cm)

    # ==================================================
    # PRECISION
    # ==================================================

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    print("\n--- Precision ---")
    print("Precision:", round(precision, 3))

    # ==================================================
    # RECALL
    # ==================================================

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    print("\n--- Recall ---")
    print("Recall:", round(recall, 3))

    # ==================================================
    # F1 SCORE
    # ==================================================

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    print("\n--- F1 Score ---")
    print("F1 Score:", round(f1, 3))

    # ==================================================
    # ROC-AUC
    # ==================================================

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    print("\n--- ROC-AUC ---")
    print("ROC-AUC:", round(roc_auc, 3))

    # ==================================================
    # CROSS VALIDATION
    # ==================================================

    cv_scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="accuracy"
    )

    print("\n--- 5-Fold Cross Validation ---")
    print("Fold Scores:", np.round(cv_scores, 3))
    print(
        "Average CV Score:",
        round(cv_scores.mean(), 3)
    )

    # ==================================================
    # CONFUSION MATRIX GRAPH
    # ==================================================

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Fail", "Pass"]
    )

    display.plot()

    plt.title("Confusion Matrix")
    plt.show()

    # ==================================================
    # FINAL SUMMARY
    # ==================================================

    print("\n--- Evaluation Summary ---")
    print("Accuracy :", round(accuracy, 3))
    print("Precision:", round(precision, 3))
    print("Recall   :", round(recall, 3))
    print("F1 Score :", round(f1, 3))
    print("ROC-AUC  :", round(roc_auc, 3))
    print("CV Score :", round(cv_scores.mean(), 3))