import numpy as np


class SGD:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate

    def update(self, parameter, gradient):
        return parameter - self.learning_rate * gradient


class Adam:
    def __init__(
        self,
        learning_rate=0.01,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8
    ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Separate optimizer state for every parameter
        self.states = {}

    def update(self, parameter, gradient, key="default"):
        if key not in self.states:
            self.states[key] = {
                "m": np.zeros_like(parameter, dtype=float),
                "v": np.zeros_like(parameter, dtype=float),
                "t": 0
            }

        state = self.states[key]
        state["t"] += 1
        t = state["t"]

        state["m"] = (
            self.beta1 * state["m"]
            + (1 - self.beta1) * gradient
        )

        state["v"] = (
            self.beta2 * state["v"]
            + (1 - self.beta2) * (gradient ** 2)
        )

        m_hat = state["m"] / (1 - self.beta1 ** t)
        v_hat = state["v"] / (1 - self.beta2 ** t)

        return parameter - (
            self.learning_rate * m_hat
            / (np.sqrt(v_hat) + self.epsilon)
        )


if __name__ == "__main__":
    weights = np.array([2.0, 3.0])
    gradients = np.array([0.5, -0.5])

    sgd = SGD(learning_rate=0.1)
    adam = Adam(learning_rate=0.1)

    print("Original Weights:", weights)
    print("SGD Updated Weights:", sgd.update(weights, gradients))
    print("Adam Updated Weights:", adam.update(weights, gradients))
