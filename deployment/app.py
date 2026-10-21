from flask import Flask, render_template, request
import joblib
import numpy as np

from src.neural_network import NeuralNetwork

app = Flask(__name__)

# Load the saved model parameters
model_data = joblib.load("models/xor_model.joblib")

model = NeuralNetwork(
    input_size=2,
    hidden_size=8,
    output_size=1
)

model.W1 = model_data["W1"]
model.b1 = model_data["b1"]
model.W2 = model_data["W2"]
model.b2 = model_data["b2"]


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None
    error = None
    input1 = ""
    input2 = ""

    if request.method == "POST":
        input1 = request.form.get("input1", "").strip()
        input2 = request.form.get("input2", "").strip()

        try:
            x1 = int(input1)
            x2 = int(input2)

            if x1 not in (0, 1) or x2 not in (0, 1):
                raise ValueError("Enter only 0 or 1.")

            X = np.array([[x1, x2]], dtype=float)
            probability = float(model.forward(X)[0, 0])
            prediction = int(probability >= 0.5)

        except ValueError:
            error = "Please enter either 0 or 1 in both fields."

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        error=error,
        input1=input1,
        input2=input2
    )


if __name__ == "__main__":
    app.run(debug=True)
