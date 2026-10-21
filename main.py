import os
import joblib
import numpy as np

from src.neural_network import NeuralNetwork

# XOR gate training data
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

# Create and train the neural network
model = NeuralNetwork(
    input_size=2,
    hidden_size=8,
    output_size=1,
    learning_rate=0.01
)

print("Training Neural Network...")
model.fit(X, y, epochs=5000)

# Create models directory if it does not exist
os.makedirs("models", exist_ok=True)

# Save trained model parameters
model_data = {
    "W1": model.W1,
    "b1": model.b1,
    "W2": model.W2,
    "b2": model.b2
}

joblib.dump(model_data, "models/xor_model.joblib")
print("\nModel saved successfully to models/xor_model.joblib")

# Load the saved model
loaded_data = joblib.load("models/xor_model.joblib")

loaded_model = NeuralNetwork(
    input_size=2,
    hidden_size=8,
    output_size=1
)

loaded_model.W1 = loaded_data["W1"]
loaded_model.b1 = loaded_data["b1"]
loaded_model.W2 = loaded_data["W2"]
loaded_model.b2 = loaded_data["b2"]

# Test loaded model
print("\nPredictions using loaded model:")
print(loaded_model.predict(X).ravel())

print("\nFinal probabilities:")
print(np.round(loaded_model.forward(X).ravel(), 4))
