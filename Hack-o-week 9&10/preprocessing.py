import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def preprocessing_demo():

    print("\n" + "=" * 60)
    print("              FEATURE ENGINEERING & SCALING")
    print("=" * 60)

    # Sample student dataset
    data = {
        "Study_Hours": [2, 4, 5, np.nan, 7, 8, 3, 6, np.nan, 9],
        "Attendance": [55, 65, 72, 80, np.nan, 90, 60, 85, 75, 95],
        "Assignments": [2, 4, 5, 6, 7, np.nan, 3, 6, 5, 9],
        "Marks": [35, 48, 55, 62, 70, 82, 42, 68, 58, 90]
    }

    df = pd.DataFrame(data)

    print("\n--- Original Dataset ---")
    print(df)

    # ==================================================
    # MISSING DATA HANDLING
    # ==================================================

    print("\n--- Missing Values Before Handling ---")
    print(df.isnull().sum())

    imputer = SimpleImputer(strategy="mean")

    df[["Study_Hours", "Attendance", "Assignments"]] = imputer.fit_transform(
        df[["Study_Hours", "Attendance", "Assignments"]]
    )

    print("\n--- Missing Values After Handling ---")
    print(df.isnull().sum())

    # ==================================================
    # FEATURE ENGINEERING
    # ==================================================

    # Create a new feature
    df["Performance_Score"] = (
        df["Study_Hours"] * 5
        + df["Attendance"] * 0.3
        + df["Assignments"] * 2
    )

    print("\n--- After Feature Engineering ---")
    print(df)

    # ==================================================
    # FEATURE SCALING
    # ==================================================

    scaler = StandardScaler()

    features = [
        "Study_Hours",
        "Attendance",
        "Assignments",
        "Performance_Score"
    ]

    df[features] = scaler.fit_transform(df[features])

    print("\n--- After Feature Scaling ---")
    print(df.round(2))

    print("\nPreprocessing completed successfully.")