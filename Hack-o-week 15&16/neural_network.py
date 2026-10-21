
import numpy as np

from src.activations import sigmoid
from src.losses import mse
from src.optimizers import Adam


class NeuralNetwork:
    def __init__(
        self,
        input_size,
        hidden_size=8,
        output_size=1,
        learning_rate=0.01
    ):
        np.random.seed(42)

        # Input layer to hidden layer
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))

        # Hidden layer to output layer
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))

        self.optimizer = Adam(learning_rate=learning_rate)

    # Forward propagation
    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = np.maximum(0, self.Z1)  # ReLU

        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = sigmoid(self.Z2)

        return self.A2

    # Backpropagation
    def backward(self, X, y, predictions):
        m = X.shape[0]

        # Output layer gradients for sigmoid + MSE
        dZ2 = (
            (predictions - y)
            * predictions
            * (1 - predictions)
        )

        dW2 = (self.A1.T @ dZ2) / m
        db2 = np.mean(dZ2, axis=0, keepdims=True)

        # Hidden layer gradients
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (self.Z1 > 0)

        dW1 = (X.T @ dZ1) / m
        db1 = np.mean(dZ1, axis=0, keepdims=True)

        # Update each parameter using its own Adam state
        self.W1 = self.optimizer.update(self.W1, dW1, key="W1")
        self.b1 = self.optimizer.update(self.b1, db1, key="b1")
        self.W2 = self.optimizer.update(self.W2, dW2, key="W2")
        self.b2 = self.optimizer.update(self.b2, db2, key="b2")

    # Train the neural network
    def fit(self, X, y, epochs=5000):
        for epoch in range(epochs):
            predictions = self.forward(X)
            loss = mse(y, predictions)

            self.backward(X, y, predictions)

            if epoch % 1000 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")

        return self

    # Predict class labels
    def predict(self, X):
        probabilities = self.forward(X)
        return (probabilities >= 0.5).astype(int)


if __name__ == "__main__":
    # XOR gate dataset
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=float)

    y = np.array([
        [0],
        [1],
        [1],
        [0]
    ], dtype=float)

    model = NeuralNetwork(
        input_size=2,
        hidden_size=8,
        output_size=1,
        learning_rate=0.01
    )

    model.fit(X, y, epochs=5000)

    print("\nXOR Gate Predictions:")
    print(model.predict(X).ravel())

    print("\nFinal Probabilities:")
    print(np.round(model.forward(X).ravel(), 4))
