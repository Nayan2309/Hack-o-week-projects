import numpy as np
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.perceptron import Perceptron
from src.activations import sigmoid, relu, tanh, step
from src.neural_network import NeuralNetwork


print("=" * 60)
print("        HACK-O-WEEK: NEURAL NETWORK LAB")
print("=" * 60)


# ============================================================
# PART 1: MODEL DEPLOYMENT USING JOBLIB
# ============================================================

print("\n[1] MODEL DEPLOYMENT USING JOBLIB")
print("-" * 50)

# Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create ML model
model = LogisticRegression(max_iter=200)

# Train model
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save model
joblib.dump(
    model,
    "models/iris_model.joblib"
)

print("Model saved successfully!")
print("File: models/iris_model.joblib")


# Load saved model
loaded_model = joblib.load(
    "models/iris_model.joblib"
)

# Test loaded model
sample = X_test[0].reshape(1, -1)

prediction = loaded_model.predict(sample)

print(
    "Prediction from saved model:",
    iris.target_names[prediction[0]]
)


# ============================================================
# PART 2: PERCEPTRON
# ============================================================

print("\n[2] PERCEPTRON")
print("-" * 50)

# AND gate dataset
X_and = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y_and = np.array([
    0,
    0,
    0,
    1
])

# Create perceptron
perceptron = Perceptron(
    learning_rate=0.1,
    epochs=20
)

# Train
perceptron.fit(
    X_and,
    y_and
)

# Predict
predictions = perceptron.predict(
    X_and
)

print("AND Gate Results:")

for inputs, prediction in zip(
    X_and,
    predictions
):

    print(
        f"Input: {inputs} -> Output: {prediction}"
    )


# ============================================================
# PART 3: ACTIVATION FUNCTIONS
# ============================================================

print("\n[3] ACTIVATION FUNCTIONS")
print("-" * 50)

values = np.array([
    -2,
    -1,
    0,
    1,
    2
])

print("Input values:")
print(values)

print("\nStep Activation:")
print(step(values))

print("\nSigmoid Activation:")
print(np.round(sigmoid(values), 4))

print("\nReLU Activation:")
print(relu(values))

print("\nTanh Activation:")
print(np.round(tanh(values), 4))


# ============================================================
# PART 4: NEURAL NETWORK FROM SCRATCH
# ============================================================

print("\n[4] NEURAL NETWORK FROM SCRATCH")
print("-" * 50)

# XOR dataset
X_xor = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

y_xor = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=float)


print("\nDataset:")
print("XOR Logic Gate")

print("\nTraining Neural Network using Adam...")

# Create neural network
nn = NeuralNetwork(
    input_size=2,
    hidden_size=4,
    output_size=1,
    optimizer="adam",
    learning_rate=0.1
)

# Train network
nn.train(
    X_xor,
    y_xor,
    epochs=5000
)


# ============================================================
# PREDICTIONS
# ============================================================

predictions = nn.predict(
    X_xor
)

probabilities = nn.predict_proba(
    X_xor
)

print("\nXOR Predictions:")
print("-" * 50)

for inputs, probability, prediction in zip(
    X_xor,
    probabilities,
    predictions
):

    print(
        f"Input: {inputs.astype(int)} | "
        f"Probability: {probability[0]:.4f} | "
        f"Prediction: {prediction[0]}"
    )


# Calculate accuracy
accuracy = np.mean(
    predictions == y_xor
) * 100

print(
    f"\nNeural Network Accuracy: {accuracy:.2f}%"
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("       ALL HACK-O-WEEK TASKS COMPLETED")
print("=" * 60)

print("\nTopics Covered:")
print("1. Model Saving using Joblib")
print("2. Model Loading and Prediction")
print("3. Perceptron")
print("4. Activation Functions")
print("5. Forward Propagation")
print("6. Backward Propagation")
print("7. Loss Function")
print("8. SGD Optimizer")
print("9. Adam Optimizer")
print("10. Neural Network using NumPy")
print("11. XOR Classification")
print("=" * 60)
