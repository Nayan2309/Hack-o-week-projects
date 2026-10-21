import numpy as np

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-8
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(
        y_true * np.log(y_pred)
        + (1 - y_true) * np.log(1 - y_pred)
    )

if __name__ == "__main__":
    actual = np.array([0, 1, 1, 0])
    predicted = np.array([0.1, 0.9, 0.8, 0.2])

    print("Loss Functions Test")
    print("Actual:", actual)
    print("Predicted:", predicted)
    print("Mean Squared Error:", round(mse(actual, predicted), 4))
    print("Binary Cross-Entropy:", round(binary_cross_entropy(actual, predicted), 4))
