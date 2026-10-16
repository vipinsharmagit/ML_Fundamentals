import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


class LinearRegressionFromScratch:
    def __init__(self, learning_rate=0.05, iterations=5000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.theta = None
        self.cost_history = []

    # Hypothesis: h(X) = Xθ
    def predict(self, X):
        X_with_bias = np.c_[np.ones(X.shape[0]), X]
        return X_with_bias @ self.theta

    # Mean Squared Error cost function
    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    # Gradient descent training
    def fit(self, X, y):
        X_with_bias = np.c_[np.ones(X.shape[0]), X]
        rows, columns = X_with_bias.shape

        self.theta = np.zeros(columns)

        for _ in range(self.iterations):
            predictions = X_with_bias @ self.theta
            errors = predictions - y

            gradient = (2 / rows) * (X_with_bias.T @ errors)
            self.theta -= self.learning_rate * gradient

            cost = self.mse(y, predictions)
            self.cost_history.append(cost)

        return self


def manual_r2(y_true, y_pred):
    total_sum_squares = np.sum((y_true - np.mean(y_true)) ** 2)
    residual_sum_squares = np.sum((y_true - y_pred) ** 2)
    return 1 - (residual_sum_squares / total_sum_squares)


def standardize(train_data, test_data):
    mean = np.mean(train_data, axis=0)
    std = np.std(train_data, axis=0)

    # Avoid division by zero
    std[std == 0] = 1

    train_scaled = (train_data - mean) / std
    test_scaled = (test_data - mean) / std

    return train_scaled, test_scaled


# 1. Load real built-in Diabetes dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Standardize features using training-data statistics only
X_train_scaled, X_test_scaled = standardize(X_train, X_test)

# 4. Train multi-feature regression model from scratch
model = LinearRegressionFromScratch(learning_rate=0.05, iterations=5000)
model.fit(X_train_scaled, y_train)

# 5. Evaluate our model manually
our_predictions = model.predict(X_test_scaled)
our_mse = model.mse(y_test, our_predictions)
our_r2 = manual_r2(y_test, our_predictions)

print("----- NumPy From-Scratch Model -----")
print(f"Test MSE: {our_mse:.4f}")
print(f"Test R²:  {our_r2:.4f}")
print(f"Intercept: {model.theta[0]:.4f}")
print("Coefficients:", model.theta[1:])

# 6. sklearn sanity-check comparison
sklearn_model = LinearRegression()
sklearn_model.fit(X_train_scaled, y_train)

sklearn_predictions = sklearn_model.predict(X_test_scaled)
sklearn_mse = np.mean((y_test - sklearn_predictions) ** 2)
sklearn_r2 = manual_r2(y_test, sklearn_predictions)

print("\n----- scikit-learn Comparison -----")
print(f"Test MSE: {sklearn_mse:.4f}")
print(f"Test R²:  {sklearn_r2:.4f}")
print(f"Intercept: {sklearn_model.intercept_:.4f}")
print("Coefficients:", sklearn_model.coef_)

# 7. Plot gradient-descent convergence
plt.figure(figsize=(8, 5))
plt.plot(model.cost_history, color="blue")
plt.title("Gradient Descent Convergence")
plt.xlabel("Iteration")
plt.ylabel("Training MSE Cost")
plt.grid(True)
plt.tight_layout()
plt.savefig("cost_convergence.png")
plt.show()

# 8. Simple linear regression example using one feature
X_simple = X[:, [2]]  # One feature only
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_simple, y, test_size=0.2, random_state=42
)

X_train_s, X_test_s = standardize(X_train_s, X_test_s)

simple_model = LinearRegressionFromScratch(learning_rate=0.05, iterations=5000)
simple_model.fit(X_train_s, y_train_s)

simple_predictions = simple_model.predict(X_test_s)

print("\n----- Simple Linear Regression -----")
print(f"Test MSE: {simple_model.mse(y_test_s, simple_predictions):.4f}")
print(f"Test R²:  {manual_r2(y_test_s, simple_predictions):.4f}")