import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score


def regression_models():

    print("\n" + "=" * 55)
    print("                    REGRESSION")
    print("=" * 55)

    # Sample house-size and house-price data
    X = np.array([
        [600], [800], [1000], [1200], [1400],
        [1600], [1800], [2000], [2200], [2400],
        [2600], [2800], [3000], [3200], [3400]
    ])

    y = np.array([
        30, 40, 50, 58, 67,
        75, 83, 92, 101, 110,
        120, 128, 138, 148, 158
    ])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ==================================================
    # LINEAR REGRESSION
    # ==================================================

    linear_model = LinearRegression()

    linear_model.fit(X_train, y_train)

    linear_prediction = linear_model.predict(X_test)

    linear_r2 = r2_score(y_test, linear_prediction)
    linear_mse = mean_squared_error(y_test, linear_prediction)

    print("\n--- Linear Regression ---")
    print("R² Score:", round(linear_r2, 3))
    print("Mean Squared Error:", round(linear_mse, 3))

    # ==================================================
    # POLYNOMIAL REGRESSION
    # ==================================================

    polynomial_model = make_pipeline(
        PolynomialFeatures(degree=2),
        LinearRegression()
    )

    polynomial_model.fit(X_train, y_train)

    polynomial_prediction = polynomial_model.predict(X_test)

    polynomial_r2 = r2_score(y_test, polynomial_prediction)
    polynomial_mse = mean_squared_error(
        y_test,
        polynomial_prediction
    )

    print("\n--- Polynomial Regression ---")
    print("R² Score:", round(polynomial_r2, 3))
    print("Mean Squared Error:", round(polynomial_mse, 3))

    # ==================================================
    # RIDGE REGRESSION
    # ==================================================

    ridge_model = Ridge(alpha=1.0)

    ridge_model.fit(X_train, y_train)

    ridge_prediction = ridge_model.predict(X_test)

    ridge_r2 = r2_score(y_test, ridge_prediction)
    ridge_mse = mean_squared_error(y_test, ridge_prediction)

    print("\n--- Ridge Regression ---")
    print("R² Score:", round(ridge_r2, 3))
    print("Mean Squared Error:", round(ridge_mse, 3))

    # ==================================================
    # LASSO REGRESSION
    # ==================================================

    lasso_model = Lasso(alpha=0.1)

    lasso_model.fit(X_train, y_train)

    lasso_prediction = lasso_model.predict(X_test)

    lasso_r2 = r2_score(y_test, lasso_prediction)
    lasso_mse = mean_squared_error(y_test, lasso_prediction)

    print("\n--- Lasso Regression ---")
    print("R² Score:", round(lasso_r2, 3))
    print("Mean Squared Error:", round(lasso_mse, 3))

    # ==================================================
    # SAMPLE PREDICTION
    # ==================================================

    house_size = np.array([[2500]])

    predicted_price = linear_model.predict(house_size)

    print("\n--- Sample Prediction ---")
    print("House Size:", house_size[0][0], "sq.ft")
    print(
        "Predicted Price:",
        round(predicted_price[0], 2),
        "Lakh"
    )

    # ==================================================
    # GRAPH
    # ==================================================

    X_line = np.linspace(
        X.min(),
        X.max(),
        100
    ).reshape(-1, 1)

    y_line = linear_model.predict(X_line)

    plt.figure(figsize=(8, 5))

    plt.scatter(
        X,
        y,
        label="Actual Data"
    )

    plt.plot(
        X_line,
        y_line,
        label="Linear Regression"
    )

    plt.xlabel("House Size (sq.ft)")
    plt.ylabel("Price (Lakh)")
    plt.title("Linear Regression - House Price Prediction")

    plt.legend()
    plt.grid()

    plt.show()