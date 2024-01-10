import numpy as np
import matplotlib.pyplot as plt

# Generate some random data for a linear relationship
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Add a bias term to X
X_b = np.c_[np.ones((100, 1)), X]

# Set hyperparameters
learning_rate = 0.01
n_iterations = 1000

# Initialize random weights
theta = np.random.randn(2, 1)

# Stochastic Gradient Descent
for iteration in range(n_iterations):
    for i in range(100):
        random_index = np.random.randint(100)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        theta = theta - learning_rate * gradients

# Print the final theta values
print("Final theta values:", theta)

# Plot the data and the linear regression line
plt.scatter(X, y)
plt.plot(X, X_b.dot(theta), color='red')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with Stochastic Gradient Descent')
plt.show()
