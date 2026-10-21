
import numpy as np


# 1. Mean Squared Error
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# Derivative of Mean Squared Error
def mse_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / len(y_true)


# 2. Binary Cross-Entropy Loss
def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-8

    y_pred = np.clip(
        y_pred,
        epsilon,
        1 - epsilon
    )

    return -np.mean(
        y_true * np.log(y_pred)
        + (1 - y_true) * np.log(1 - y_pred)
    )


# Test the loss functions
if __name__ == "__main__":

    actual = np.array([[0], [1], [1], [0]])
    predicted = np.array([[0.1], [0.9], [0.8], [0.2]])

    print("Actual Values:")
    print(actual.ravel())

    print("\nPredicted Values:")
    print(predicted.ravel())

    print("\nMean Squared Error:")
    print(round(mse(actual, predicted), 4))

    print("\nBinary Cross-Entropy:")
    print(round(binary_cross_entropy(actual, predicted), 4))
