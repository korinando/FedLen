import numpy as np


# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# One-vs-Rest Logistic Regression Classifier
class OneVsRestLR:
    def __init__(self, learning_rate=0.01, n_iterations=100):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.params_dict = dict()
        self.classes = list()

    def _binary_logistic_regression(self, X, y, weights, bias):
        """Trains a binary logistic regression classifier for one class."""
        num_samples, num_features = X.shape

        # Gradient descent
        for i in range(self.n_iterations):
            # Linear model
            linear_model = np.dot(X, weights) + bias
            # Sigmoid activation
            y_predicted = sigmoid(linear_model)

            # Compute gradients
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)

            # Update weights and bias
            weights -= self.learning_rate * dw
            bias -= self.learning_rate * db

        return weights, bias

    def fit(self, X, y):
        """Trains One-vs-Rest logistic regression for multi-class classification."""
        # Extract unique classes from the labels
        self.classes = np.unique(y)

        # Train one binary logistic regression model per class
        for c in self.classes:
            # Create binary labels (1 for the current class, 0 for others)
            binary_y = np.where(y == c, 1, 0)

            # Initialize weights and bias if not provided in params_dict
            if c not in self.params_dict:
                self.params_dict[c] = {
                    'weights': np.zeros(X.shape[1]),
                    'bias': 0
                }

            weights = self.params_dict[c]['weights']
            bias = self.params_dict[c]['bias']

            # Train binary logistic regression
            weights, bias = self._binary_logistic_regression(X, binary_y, weights, bias)

            # Store weights and bias in the dictionary with the class label as key
            self.params_dict[c] = {
                'weights': weights,
                'bias': bias
            }

    def predict(self, X, y):
        """Predicts class labels for the input data."""
        # Extract unique classes from the labels
        self.classes = np.unique(y)

        # Collect probabilities from each binary classifier
        class_probabilities = np.zeros((X.shape[0], len(self.classes)))

        for idx, c in enumerate(self.classes):
            # Get weights and bias for the current class
            weights = self.params_dict[c]['weights']
            bias = self.params_dict[c]['bias']

            # Predict probabilities for the current class
            linear_model = np.dot(X, weights) + bias
            class_probabilities[:, idx] = sigmoid(linear_model)

        # Assign the class with the highest probability
        return self.classes[np.argmax(class_probabilities, axis=1)]
