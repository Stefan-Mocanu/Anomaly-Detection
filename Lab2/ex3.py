from sklearn.datasets import make_blobs
from pyod.models import knn, lof

import matplotlib.pyplot as plt

X, Y = make_blobs(
    n_samples=[200,100],
    centers=[[-10,-10],[10,10]],
    cluster_std=[2,6],
    random_state=42
    )

neighbors = 2

knn_model = knn.KNN(contamination=0.07, n_neighbors=neighbors)
lof_model = lof.LOF(contamination=0.07, n_neighbors=neighbors)
knn_model.fit(X)
lof_model.fit(X)
Y_pred_knn = knn_model.predict(X)
Y_pred_lof = lof_model.predict(X)
fig,axs = plt.subplots(nrows=1, ncols=2)

axs[0].scatter(X[Y_pred_knn == 0, 0], X[Y_pred_knn == 0, 1], c='blue', label='Inliers', alpha=0.6)
axs[0].scatter(X[Y_pred_knn == 1, 0], X[Y_pred_knn == 1, 1], c='red', label='Outliers', alpha=0.8)
axs[0].set_title('KNN')
# axs[0].legend()
axs[1].scatter(X[Y_pred_lof == 0, 0], X[Y_pred_lof == 0, 1], c='blue', label='Inliers', alpha=0.6)
axs[1].scatter(X[Y_pred_lof == 1, 0], X[Y_pred_lof == 1, 1], c='red', label='Outliers', alpha=0.8)
axs[1].set_title('LOF')

plt.savefig(f"Lab2/ex3_{neighbors}_neighbors.png")