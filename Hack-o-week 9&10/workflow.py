import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def sklearn_workflow():

    print("\n" + "=" * 60)
    print("             SCIKIT-LEARN WORKFLOW")
    print("=" * 60)

    # --------------------------------------------------
    # DATASET
    # --------------------------------------------------

    data = {
        "Study_Hours": [2, 4, 5, np.nan, 7, 8, 3, 6, 4, 9,
                        5, 7, 2, 8, 6, 10, 3, 9, 5, 7],

        "Attendance": [55, 65, 72, 80, np.nan, 90, 60, 85, 66, 95,
                       75, 82, 58, 88, 79, 97, 62, 93, 70, 84],

        "Assignments": [2, 4, 5, 6, 7, np.nan, 3, 6, 4, 9,
                        5, 7, 2, 8, 6, 10, 3, 9, 5, 7],

        "Pass": [0, 0, 0, 1, 1, 1, 0, 1, 0, 1,
                 1, 1, 0, 1, 1, 1, 0, 1, 0, 1]
    }

    df = pd.DataFrame(data)

    print("\n--- Original Dataset ---")
    print(df)

    # --------------------------------------------------
    # FEATURES AND TARGET
    # --------------------------------------------------

    X = df[["Study_Hours", "Attendance", "Assignments"]]
    y = df["Pass"]

    # --------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------

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

    # --------------------------------------------------
    # SCIKIT-LEARN PIPELINE
    # --------------------------------------------------

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression())
    ])

    # --------------------------------------------------
    # MODEL BUILDING
    # --------------------------------------------------

    print("\n--- Building Model ---")

    pipeline.fit(X_train, y_train)

    print("Pipeline completed successfully.")

    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------

    y_pred = pipeline.predict(X_test)

    # --------------------------------------------------
    # EVALUATION
    # --------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)

    print("\n--- Model Evaluation ---")
    print("Accuracy:", round(accuracy * 100, 2), "%")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["Fail", "Pass"],
        zero_division=0
    ))

    # --------------------------------------------------
    # SAMPLE PREDICTION
    # --------------------------------------------------

    student = pd.DataFrame({
        "Study_Hours": [6],
        "Attendance": [80],
        "Assignments": [6]
    })

    prediction = pipeline.predict(student)[0]
    probability = pipeline.predict_proba(student)[0][1]

    print("\n--- New Student Prediction ---")
    print("Study Hours:", student["Study_Hours"].iloc[0])
    print("Attendance:", student["Attendance"].iloc[0])
    print("Assignments:", student["Assignments"].iloc[0])

    print(
        "Prediction:",
        "PASS" if prediction == 1 else "FAIL"
    )

    print(
        "Pass Probability:",
        round(probability * 100, 2),
        "%"
    )

    # --------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------

    print("\n--- Model Interpretation ---")

    model = pipeline.named_steps["model"]

    feature_names = [
        "Study Hours",
        "Attendance",
        "Assignments"
    ]

    coefficients = model.coef_[0]

    for feature, coefficient in zip(
        feature_names,
        coefficients
    ):
        if coefficient > 0:
            effect = "positive"
        else:
            effect = "negative"

        print(
            feature,
            "->",
            round(coefficient, 3),
            "(" + effect + " effect)"
        )

    print("\nWorkflow:")
    print("Data")
    print("  ↓")
    print("Missing Value Handling")
    print("  ↓")
    print("Feature Scaling")
    print("  ↓")
    print("Logistic Regression")
    print("  ↓")
    print("Prediction")
    print("  ↓")
    print("Evaluation")
    print("  ↓")
    print("Interpretation")