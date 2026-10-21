import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


def calculus():
    print("\n" + "=" * 50)
    print("                 CALCULUS")
    print("=" * 50)

    # ---------------- DERIVATIVE ----------------
    print("\n--- Derivative ---")

    x = sp.symbols('x')

    function = x**2 + 3*x + 2
    derivative = sp.diff(function, x)

    print("Function: f(x) =", function)
    print("Derivative: f'(x) =", derivative)

    # ---------------- GRADIENT ----------------
    print("\n--- Gradient ---")

    y = sp.symbols('y')

    function_2 = x**2 + y**2

    gradient_x = sp.diff(function_2, x)
    gradient_y = sp.diff(function_2, y)

    print("Function: f(x,y) =", function_2)
    print("∂f/∂x =", gradient_x)
    print("∂f/∂y =", gradient_y)
    print("Gradient =", [gradient_x, gradient_y])

    # ---------------- CHAIN RULE ----------------
    print("\n--- Chain Rule ---")

    u = sp.symbols('u')

    outer_function = u**2
    inner_function = 3*x + 1

    dy_du = sp.diff(outer_function, u)
    du_dx = sp.diff(inner_function, x)

    chain_rule = dy_du.subs(u, inner_function) * du_dx

    print("Outer function: y = u²")
    print("Inner function: u =", inner_function)
    print("dy/du =", dy_du)
    print("du/dx =", du_dx)
    print("Using Chain Rule:")
    print("dy/dx =", sp.expand(chain_rule))

    # ---------------- BACKPROPAGATION ----------------
    print("\n--- Backpropagation Example ---")

    input_value = 2
    weight = 0.5
    target = 2

    # Forward pass
    output = input_value * weight

    # Error
    error = output - target

    # Gradient using chain rule
    gradient = error * input_value

    # Learning rate
    learning_rate = 0.1

    # Weight update
    new_weight = weight - learning_rate * gradient

    print("Input:", input_value)
    print("Initial Weight:", weight)
    print("Target:", target)
    print("Output:", output)
    print("Error:", error)
    print("Gradient:", gradient)
    print("Updated Weight:", round(new_weight, 2))

    print("\nBackpropagation Flow:")
    print("Input → Output → Error → Gradient → Weight Update")

    # ---------------- DERIVATIVE GRAPH ----------------
    print("\n--- Derivative Graph ---")

    x_values = np.linspace(-5, 5, 100)

    function_values = x_values**2 + 3*x_values + 2
    derivative_values = 2*x_values + 3

    plt.figure(figsize=(8, 5))

    plt.plot(
        x_values,
        function_values,
        label="f(x) = x² + 3x + 2"
    )

    plt.plot(
        x_values,
        derivative_values,
        label="f'(x) = 2x + 3"
    )

    plt.xlabel("x")
    plt.ylabel("Value")
    plt.title("Function and Its Derivative")

    plt.legend()
    plt.grid()

    plt.show()