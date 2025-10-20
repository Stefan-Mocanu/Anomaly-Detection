from pyod.utils.data import generate_data 
import matplotlib.pyplot as plt
import numpy as np

X_train, X_test, y_train, y_test = generate_data(400,100,2,0.1)

X = np.concatenate([X_train, X_test])
y = np.concatenate([y_train, y_test])

plt.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', label='Inliers', alpha=0.6)
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', label='Outliers', alpha=0.8)
plt.legend()
plt.title("Generated Data with Outliers")
plt.savefig("Lab1/ex1.png")