import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error


def bias_variance():

    print("\n" + "=" * 60)
    print("             BIAS-VARIANCE TRADE-OFF")
    print("=" * 60)

    # Generate data
    np.random.seed(42)

    X = np.linspace(0, 10, 30)
    y = 2 * X + 3 + np.random.normal(0, 3, 30)

    X = X.reshape(-1, 1)

    degrees = [1, 2, 5, 10]

    errors = []

    print("\nModel Complexity and Error:")
    print("-" * 40)

    for degree in degrees:

        model = make_pipeline(
            PolynomialFeatures(degree),
            LinearRegression()
        )

        model.fit(X, y)

        predictions = model.predict(X)

        mse = mean_squared_error(y, predictions)

        errors.append(mse)

        print(
            f"Polynomial Degree {degree:<2} "
            f"-> MSE: {mse:.2f}"
        )

    # Plot
    plt.figure(figsize=(8, 5))

    plt.plot(
        degrees,
        errors,
        marker="o"
    )

    plt.xlabel("Model Complexity")
    plt.ylabel("Mean Squared Error")
    plt.title("Bias-Variance Trade-off")

    plt.grid()
    plt.tight_layout()
    plt.show()

    print("\n--- Interpretation ---")

    print("High bias usually occurs when a model is too simple.")
    print("High variance usually occurs when a model is too complex.")
    print("The goal is to find a suitable balance between bias and variance.")

    print("\nBias-Variance trade-off completed successfully.")