import numpy as np
import matplotlib.pyplot as plt

# Sigmoid activation function
def sigmoid(u):
    return 1 / (1 + np.exp(-u))

# Derivative of sigmoid
def sigmoid_derivative(u):
    return u * (1 - u)

# Compute error function
def compute_error(y, t):
    return np.sum((y - t) ** 2)

# Two-layer Neural Network Training with Backpropagation
def train_two_layer_network(X, t, learning_rate=0.5, epochs=1000):
    np.random.seed(0)
    input_size = X.shape[1]  # Number of input neurons
    hidden_size = 2  # Number of hidden neurons (adjustable)
    output_size = 1  # Single output neuron

    # Initialize weights randomly
    w_input_hidden = np.random.rand(input_size, hidden_size)
    w_hidden_output = np.random.rand(hidden_size, output_size)

    errors = []
    w_input_hidden_history = np.zeros((epochs, input_size, hidden_size))
    w_hidden_output_history = np.zeros((epochs, hidden_size, output_size))

    for epoch in range(epochs):
        # Forward pass
        z_hidden = sigmoid(np.dot(X, w_input_hidden))  # Hidden layer activations
        y_output = sigmoid(np.dot(z_hidden, w_hidden_output))  # Output layer activations

        # Compute error
        error = compute_error(y_output, t.reshape(-1, 1))
        errors.append(error)

        # Backpropagation
        output_error = (y_output - t.reshape(-1, 1)) * sigmoid_derivative(y_output)  # Output layer delta
        hidden_error = np.dot(output_error, w_hidden_output.T) * sigmoid_derivative(z_hidden)  # Hidden layer delta

        # Update weights
        w_hidden_output -= learning_rate * np.dot(z_hidden.T, output_error)
        w_input_hidden -= learning_rate * np.dot(X.T, hidden_error)

        # Store weight history
        w_input_hidden_history[epoch] = w_input_hidden
        w_hidden_output_history[epoch] = w_hidden_output

        # Print error at intervals for monitoring
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Error = {error:.6f}")

    return w_input_hidden, w_hidden_output, errors, w_input_hidden_history, w_hidden_output_history, y_output

# Training Data (Sample Set from HW3)
X = np.array([[0.1, 0.1, 1],
              [0.1, 0.9, 1],
              [0.9, 0.1, 1],
              [0.9, 0.9, 1]])

t = np.array([0.1, 0.9, 0.9, 0.1])  # Target values

# Train the network
learning_rate = 0.5
epochs = 1000
w_input_hidden, w_hidden_output, errors, w_input_hidden_history, w_hidden_output_history, y_output = train_two_layer_network(X, t, learning_rate, epochs)

# Print Final Weights
print("Final Weights (Input to Hidden):\n", w_input_hidden)
print("Final Weights (Hidden to Output):\n", w_hidden_output)

# Print Final Output Values
print("Final Output Values:\n", y_output.flatten())

# Plot Error vs. Epochs (Humanized scale)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(epochs), errors, label='Error', color='red')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Error vs. Time')
plt.legend()
plt.grid(True)

# Plot Weights vs Time (Raw values without smoothing)
plt.subplot(1, 2, 2)
plt.plot(w_input_hidden_history[:, 0, 0], label='w1_hidden', color='blue')
plt.plot(w_input_hidden_history[:, 1, 0], label='w2_hidden', color='green')
plt.plot(w_input_hidden_history[:, 2, 0], label='w3_hidden', color='purple')
plt.plot(w_hidden_output_history[:, 0, 0], label='w_hidden_output', color='orange')

plt.xlabel('Epochs')
plt.ylabel('Weights')
plt.title('Weights vs. Time')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

