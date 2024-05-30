import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, num_iterations=10):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        y_ = np.where(y <= 0, -1, 1)

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            for idx, x_i in enumerate(X):
                calcualted_value = y_[idx] * (np.dot(x_i, self.weights) - self.bias)
                print(f"Weights : {self.weights} , Class : {y_[idx]} , Value :{x_i} , Out: {calcualted_value}")
                condition = y_[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights - np.dot(x_i, y_[idx]))
                    self.bias -= self.lr * y_[idx]

    def predict(self, X):
        approximated = np.dot(X, self.weights) - self.bias
        return np.sign(approximated)

# Example usage:
if __name__ == "__main__":
    # Sample data
    X = np.array([[1, 2], [5, 8], [1.5, 1.8], [8, 8], [1, 0.6], [9, 11]])
    y = np.array([-1, 1, -1, 1, -1, 1])

    # Create and train SVM model
    svm = SVM()
    svm.fit(X, y)

    # Make predictions
    print("Predictions:", svm.predict(X))