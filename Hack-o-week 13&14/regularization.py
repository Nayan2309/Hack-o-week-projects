import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score


def regularization():

    print("\n" + "=" * 60)
    print("          REGULARIZATION")
    print("=" * 60)

    # Dataset
    diabetes = load_diabetes()

    X = diabetes.data
    y = diabetes.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    models = {

        "Linear Regression": make_pipeline(
            StandardScaler(),
            LinearRegression()
        ),

        "L1 - Lasso": make_pipeline(
            StandardScaler(),
            Lasso(alpha=0.1)
        ),

        "L2 - Ridge": make_pipeline(
            StandardScaler(),
            Ridge(alpha=1.0)
        )
    }

    results = {}

    print("\nModel Performance:")
    print("-" * 50)

    for name, model in models.items():

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        results[name] = r2

        print(f"\n{name}")
        print(f"MSE : {mse:.2f}")
        print(f"R²  : {r2:.3f}")

    # Graph
    plt.figure(figsize=(9, 5))

    plt.bar(
        results.keys(),
        results.values()
    )

    plt.xlabel("Model")
    plt.ylabel("R² Score")
    plt.title("Regularization Comparison")

    plt.ylim(0, 1)
    plt.grid(axis="y")

    plt.tight_layout()
    plt.show()

    print("\n--- Interpretation ---")

    print("L1 regularization is also called Lasso.")
    print("L1 can reduce some feature coefficients to zero.")
    print("L2 regularization is also called Ridge.")
    print("L2 reduces large coefficients without usually making them zero.")
    print("Regularization helps control overfitting and improves generalization.")

    print("\nRegularization completed successfully.")