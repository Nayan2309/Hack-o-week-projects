
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def step(x):
    return np.where(x >= 0, 1, 0)


if __name__ == "__main__":
    values = np.array([-2, -1, 0, 1, 2])

    print("Input:", values)
    print("Sigmoid:", np.round(sigmoid(values), 4))
    print("ReLU:", relu(values))
    print("Tanh:", np.round(tanh(values), 4))
    print("Step:", step(values))
