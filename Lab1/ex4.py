from pyod.utils.data import generate_data 
from sklearn.metrics import balanced_accuracy_score
import numpy as np

N_TOTAL = 1000
CONTAMINATION_RATE = 0.1
N_OUTLIERS = int(N_TOTAL * CONTAMINATION_RATE)
N_NORMAL = N_TOTAL - N_OUTLIERS

mu = np.array([3,2,6])
L = np.array([[0.5, 0.7, 2],
              [3,0.2, 0.1],
              [3,0.9,0.2]])

Sigma = L @ L.T
Sigma_inv = np.linalg.inv(Sigma)

X_normal_std = np.random.randn(N_NORMAL, 3)
Y_normal = X_normal_std @ L.T + mu

mu_outlier =  mu + 2 * np.ones(3)
L_outlier = 2 * L
X_outlier_std = np.random.randn(N_OUTLIERS, 3)
Y_outlier = X_outlier_std @ L_outlier.T + mu_outlier

Y_train = np.vstack([Y_normal, Y_outlier])
labels_train = np.hstack([np.zeros(N_NORMAL), np.ones(N_OUTLIERS)]).astype(int)

z_scores = np.array([
    np.sqrt((y - mu).T @ Sigma_inv @ (y - mu))
    for y in Y_train
])


threshold = np.quantile(z_scores, 1 - CONTAMINATION_RATE)

labels_pred = (z_scores > threshold).astype(int)

balanced_acc = balanced_accuracy_score(labels_train, labels_pred)

print("Balanced accuracy:", balanced_acc)
