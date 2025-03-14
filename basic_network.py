from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

data = load_breast_cancer()
X, y = data.data, data.target
print(f'The data has a shape of {X.shape}, and the target has a shape of {y.shape}')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'The training set has {X_train.shape[0]} datapoints and the test set has {X_test.shape[0]} datapoints.')

scaler = MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
print(f'The max of training data is {X_train.max():.2f} and the min is {X_train.min():.2f}.')

import numpy as np
# initialize numpy random seed
np.random.seed(29)

# Sigmoid function for logistic regression
def sigmoid(z):
    z_sigmoid = 1 / (1 + np.exp(-z))
    return z_sigmoid

# Binary Cross-Entropy Loss
def binary_cross_entropy(y_true, y_pred):
    # Avoid log(0) by clipping predictions
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

class LogisticRegression_SGD:
    def __init__(self, learning_rate=0.01, epochs=100, batch_size=32):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

    # Initialize weights
    def initialize_weights(self, n_features):
        """
        Initializes weights and bias to zero.

        :param n_features: Number of input features
        """
        n_features = X_train.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0


    # Prediction function
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = sigmoid(linear_model)
        return y_predicted

    # Training function using mini-batch SGD
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.initialize_weights(n_features)

        for epoch in range(self.epochs):
            # Shuffle the data
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]

            if epoch == 0:
              loss = binary_cross_entropy(y, self.predict(X))
              print("SGD loss")
              print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss:.4f}")

            for i in range(0, n_samples, self.batch_size):
                X_batch = X[i:i + self.batch_size]
                y_batch = y[i:i + self.batch_size]
                # Predictions
                y_predicted = self.predict(X_batch)

                # Compute gradients

                gradient_w = np.dot(X_batch.T, (y_predicted - y_batch)) / self.batch_size
                gradient_b = np.mean(y_predicted - y_batch)

                # Update weights
                self.weights -= self.learning_rate * gradient_w
                self.bias -= self.learning_rate * gradient_b

            # Calculate loss for monitoring
            loss = binary_cross_entropy(y, self.predict(X))
            if (epoch + 1) % 10 == 0:
              print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss:.4f}")

model_SGD = LogisticRegression_SGD(learning_rate=0.1, epochs=100, batch_size=16)
model_SGD.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

predictions = model_SGD.predict(X_test)
predictions = (predictions > 0.5).astype(int)
print(predictions)
print(y_test)
accuracy = accuracy_score(y_test, predictions)

print(f'The accuracy is {accuracy:.4f}')

def relu(z):
    """ReLU activation function."""
    z_relu = np.maximum(0, z)
    return z_relu

def relu_derivative(z):
    """Derivative of ReLU activation function."""
    z_relu_derivative = np.where(z > 0, 1, 0)
    return z_relu_derivative

def sigmoid(z):
    """Sigmoid activation function."""
    z_sigmoid = 1 / (1 + np.exp(-z))
    return z_sigmoid
  
class NeuralNetwork:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, learning_rate=0.01, epochs=100, batch_size=32):
        """
        Initialize the Neural Network with given parameters.
        :param input_size: Number of input features
        :param hidden_size1: Number of neurons in the first hidden layer
        :param hidden_size2: Number of neurons in the second hidden layer
        :param output_size: Number of output neurons (1 for binary classification)
        :param learning_rate: Learning rate for weight updates
        :param epochs: Number of training iterations
        :param batch_size: Size of mini-batches for SGD
        """
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize weights and biases using Kaiming initialization."""

        # Kaiming initialization for the first layer weights
        limit = np.sqrt(6 / self.input_size)
        self.W1 = np.random.normal(0.0, limit, size=(self.input_size, self.hidden_size1))
        self.b1 = np.zeros(self.hidden_size1)

        # Kaiming initialization for the second layer weights
        limit = np.sqrt(6 / self.hidden_size1)
        self.W2 = np.random.normal(0.0, limit, size=(self.hidden_size1, self.hidden_size2))
        self.b2 = np.zeros(self.hidden_size2)

        # Kaiming initialization for the third layer weights
        limit = np.sqrt(6 / self.hidden_size2)
        self.W3 = np.random.normal(0.0, limit, size=(self.hidden_size2, self.output_size))
        self.b3 = np.zeros(self.output_size)


    def forward(self, X):
        """
        Forward pass through the network.
        :param X: Input data
        :return: Activated output of the network
        """
        # print(self.input_size)
        self.Z1 = np.dot(X, self.W1) + self.b1  # Store pre-activation values
        self.first_layer_output = relu(self.Z1)
        # print(self.first_layer_output.shape)
        self.Z2 = np.dot(self.first_layer_output, self.W2) + self.b2  # Store pre-activation values
        self.second_layer_output = relu(self.Z2)

        self.Z3 = np.dot(self.second_layer_output, self.W3) + self.b3
        self.output = sigmoid(self.Z3)

        return self.output

    def backward(self, X, y, output):
        """
        Backpropagation to compute gradients and update weights.
        :param X: Input data
        :param y: True labels
        :param output: Predicted output from forward pass
        """
        m = X.shape[0]

        # Gradient of loss w.r.t. output (binary cross-entropy with sigmoid activation)
        dZ3 = output - y[:,None]  # Gradient wrt Z3 when using sigmoid activation at output
        # total loss basically

        # Gradients for the third (output) layer
        dW3 = np.dot(self.second_layer_output.T, dZ3) / m
        db3 = np.sum(dZ3, axis=0) / m

        # Gradients for the second hidden layer
        dZ2 = np.dot(dZ3, self.W3.T) * relu_derivative(self.Z2)  # Use pre-activation values (Z2)
        dW2 = np.dot(self.first_layer_output.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0) / m

        # Gradients for the first hidden layer
        dZ1 = np.dot(dZ2, self.W2.T) * relu_derivative(self.Z1)  # Use pre-activation values (Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0) / m

        # Update weights and biases
        self.W3 -= self.learning_rate * dW3
        self.b3 -= self.learning_rate * db3
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def fit(self, X, y):
        """
        Train the neural network using mini-batch SGD.
        :param X: Training data
        :param y: True labels
        """
        loss = binary_cross_entropy(y, self.forward(X))
        print(f"Epoch 0/{self.epochs}, Loss: {loss:.4f}")

        for epoch in range(self.epochs):
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]

            for i in range(0, X.shape[0], self.batch_size):
                X_batch = X[i:i + self.batch_size]
                y_batch = y[i:i + self.batch_size]

                # Forward and backward pass
                output = self.forward(X_batch)
                self.backward(X_batch, y_batch, output)

            # Calculate and print loss for monitoring
            if (epoch + 1) % 100 == 0:
              loss = binary_cross_entropy(y, self.forward(X))
              print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss:.4f}")

    def predict(self, X):
        """
        Predict using the trained neural network.
        :param X: Input data
        :return: Predicted labels
        """

        predictions = self.forward(X)
        return (predictions > 0.5).astype(int)  # Convert probabilities to binary predictions
      
nn_network = NeuralNetwork(input_size=X_train.shape[1], hidden_size1=8, hidden_size2=4, output_size=1, learning_rate=0.0001, epochs=1000, batch_size=16)
nn_network.fit(X_train, y_train)

predictions = nn_network.predict(X_test)
print(predictions.reshape(-1))
print(y_test)
accuracy = accuracy_score(y_test, predictions)

print(f'The accuracy is {accuracy:.4f}')
