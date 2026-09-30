import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def ensemble_methods():

    print("\n" + "=" * 60)
    print("              ENSEMBLE METHODS")
    print("=" * 60)

    # Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = {

        "Bagging": BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=42),
            n_estimators=50,
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            random_state=42
        ),

        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            eval_metric="mlogloss"
        ),

        "LightGBM": LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
            verbosity=-1
        )
    }

    results = {}

    print("\nModel Accuracy:")
    print("-" * 40)

    for name, model in models.items():

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        results[name] = accuracy

        print(f"{name:<20}: {accuracy * 100:.2f}%")

    # Graph
    plt.figure(figsize=(10, 5))

    plt.bar(
        results.keys(),
        [value * 100 for value in results.values()]
    )

    plt.xlabel("Ensemble Method")
    plt.ylabel("Accuracy (%)")
    plt.title("Ensemble Methods Accuracy Comparison")

    plt.xticks(rotation=20)
    plt.ylim(0, 110)
    plt.grid(axis="y")

    plt.tight_layout()
    plt.show()

    print("\n--- Interpretation ---")

    print("Bagging trains multiple models independently and combines their results.")
    print("Random Forest is a popular bagging-based ensemble method.")
    print("Boosting trains models sequentially to correct previous errors.")
    print("XGBoost is an optimized gradient boosting algorithm.")
    print("LightGBM is a fast and efficient gradient boosting framework.")

    print("\nEnsemble methods can improve model stability and prediction performance.")