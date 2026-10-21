
import numpy as np
from src.activations import step


class Perceptron:

    def __init__(self, learning_rate=0.1, epochs=20):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for epoch in range(self.epochs):
            for xi, target in zip(X, y):
                linear_output = np.dot(xi, self.weights) + self.bias
                prediction = step(linear_output)
                error = target - prediction

                self.weights += self.learning_rate * error * xi
                self.bias += self.learning_rate * error

        return self

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return step(linear_output)


if __name__ == "__main__":
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    y = np.array([0, 0, 0, 1])

    model = Perceptron()
    model.fit(X, y)

    print("AND Gate Predictions:")
    print(model.predict(X))
