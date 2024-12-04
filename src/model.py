import numpy as np


# # Sigmoid function
# def sigmoid(z):
#     return 1 / (1 + np.exp(-z))
#
#
# # One-vs-Rest Logistic Regression Classifier
# class OneVsRestLR:
#     def __init__(self, learning_rate=0.01, n_iterations=100):
#         self.learning_rate = learning_rate
#         self.n_iterations = n_iterations
#         self.params_dict = dict()
#         self.classes = list()
#
#     def _binary_logistic_regression(self, X, y, weights, bias):
#         num_samples, num_features = X.shape
#
#         # Gradient descent
#         for i in range(self.n_iterations):
#             # Linear model
#             linear_model = np.dot(X, weights) + bias
#             # Sigmoid activation
#             y_predicted = sigmoid(linear_model)
#
#             # Compute gradients
#             dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
#             db = (1 / num_samples) * np.sum(y_predicted - y)
#
#             # Update weights and bias
#             weights -= self.learning_rate * dw
#             bias -= self.learning_rate * db
#
#         return weights, bias
#
#     def fit(self, X, y):
#         # Extract unique classes from the labels
#         self.classes = np.unique(y)
#
#         # Train one binary logistic regression model per class
#         for c in self.classes:
#             # Create binary labels (1 for the current class, 0 for others)
#             binary_y = np.where(y == c, 1, 0)
#
#             # Initialize weights and bias if not provided in params_dict
#             if c not in self.params_dict:
#                 self.params_dict[c] = {
#                     'weights': np.zeros(X.shape[1]),
#                     'bias': 0
#                 }
#
#             weights = self.params_dict[c]['weights']
#             bias = self.params_dict[c]['bias']
#
#             # Train binary logistic regression
#             weights, bias = self._binary_logistic_regression(X, binary_y, weights, bias)
#
#             # Store weights and bias in the dictionary with the class label as key
#             self.params_dict[c] = {
#                 'weights': weights,
#                 'bias': bias
#             }
#
#     def predict(self, X, y):
#         # Extract unique classes from the labels
#         self.classes = np.unique(y)
#
#         # Collect probabilities from each binary classifier
#         class_probabilities = np.zeros((X.shape[0], len(self.classes)))
#
#         for idx, c in enumerate(self.classes):
#             # Get weights and bias for the current class
#             weights = self.params_dict[c]['weights']
#             bias = self.params_dict[c]['bias']
#
#             # Predict probabilities for the current class
#             linear_model = np.dot(X, weights) + bias
#             class_probabilities[:, idx] = sigmoid(linear_model)
#
#         # # Add Laplace noise to the predictions for differential privacy
#         # epsilon = 0.5
#         # sensitivity = 1.0  # Sensitivity of the query
#         # scale = sensitivity / epsilon
#         # noise = np.random.laplace(0, scale, class_probabilities.shape)
#         # class_probabilities += noise
#
#         # Assign the class with the highest probability
#         return self.classes[np.argmax(class_probabilities, axis=1)]
# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# One-vs-Rest Logistic Regression Classifier
class OneVsRestLR:
    def __init__(self, learning_rate=0.01, n_iterations=100):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.params_dict = {}

    def _binary_logistic_regression(self, X, y, weights, bias):
        """Trains a binary logistic regression classifier for one class with noise to gradients."""
        num_samples, num_features = X.shape
        noise_std = 0.1

        # Gradient descent
        for i in range(self.n_iterations):
            # Linear model
            linear_model = np.dot(X, weights) + bias
            # Sigmoid activation
            y_predicted = sigmoid(linear_model)

            # Compute gradients
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)

            # Add noise to gradients
            noise_dw = np.random.normal(0, noise_std, dw.shape)  # Adding noise to gradients
            noise_db = np.random.normal(0, noise_std, db.shape)  # Adding noise to bias

            # Apply noise to gradients
            dw += noise_dw
            db += noise_db

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

    def predict(self, X, y, noise_std=0.1, top_k=3):
        """Predicts class labels for the input data with noise and top-k masking."""
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

        # Add Gaussian noise to the predicted probabilities
        noise = np.random.normal(0, noise_std, class_probabilities.shape)
        class_probabilities += noise

        # Ensure probabilities remain between 0 and 1
        class_probabilities = np.clip(class_probabilities, 0, 1)

        # Mask output: Only return the top-k most likely classes
        top_k_indices = np.argsort(class_probabilities, axis=1)[:, -top_k:]

        # Create a mask for the top-k classes
        masked_probabilities = np.zeros_like(class_probabilities)
        for i, indices in enumerate(top_k_indices):
            masked_probabilities[i, indices] = class_probabilities[i, indices]

        # Get the predicted class based on the top-k masked probabilities
        predicted_classes = np.argmax(masked_probabilities, axis=1)

        return self.classes[predicted_classes]