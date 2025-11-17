from sklearn.datasets import make_blobs
from pyod.models import iforest, dif, loda
import numpy as np
import matplotlib.pyplot as plt

X, y = make_blobs(
    n_samples=[500,500], 
    n_features=3, 
    centers = [[0,10,0],[10,0,10]], 
    cluster_std=1
    )

hidden_neurons = [10,10,10]
bins = 100


modelIforest = iforest.IForest(contamination=0.02)
modelDIF = dif.DIF(contamination=0.02, hidden_neurons=hidden_neurons)
modelLODA = loda.LODA(contamination=0.02, n_bins = bins)

X_test = np.random.uniform(low=-10,high=20,size=(1000,3))

modelIforest.fit(X)
modelDIF.fit(X)
modelLODA.fit(X)

scores_test_Iforest = modelIforest.decision_function(X_test)
scores_test_DIF = modelDIF.decision_function(X_test)
scores_test_LODA = modelLODA.decision_function(X_test)

fig = plt.figure(figsize=(21, 7))
axs = [fig.add_subplot(1, 3, i+1, projection='3d') for i in range(3)]

sc = axs[0].scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=scores_test_Iforest, cmap="viridis", s=25,alpha=0.6)
plt.colorbar(sc, ax=axs[0], label="Anomaly score (higher = more normal)")

axs[0].scatter(X[:, 0], X[:, 1], X[:, 2], c="red", s=10, label="Training data",alpha=0.6)
axs[0].legend()
axs[0].set_title("Isolation Forest — Anomaly Scores and Artifacts")
# axs[0].set_xlabel("x_1")
# axs[0].set_ylabel("x_2")


sc = axs[1].scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=scores_test_DIF, cmap="viridis", s=25,alpha=0.6)
plt.colorbar(sc, ax=axs[1], label="Anomaly score (higher = more normal)")

axs[1].scatter(X[:, 0], X[:, 1], X[:, 2], c="red", s=10, label="Training data",alpha=0.6)
axs[1].legend()
axs[1].set_title("DIF")
# axs[1].set_xlabel("x_1")
# axs[1].set_ylabel("x_2")

sc = axs[2].scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=scores_test_LODA, cmap="viridis", s=25,alpha=0.6)
plt.colorbar(sc, ax=axs[2], label="Anomaly score (higher = more normal)")

axs[2].scatter(X[:, 0], X[:, 1], X[:, 2], c="red", s=10, label="Training data",alpha=0.6)
axs[2].legend()
axs[2].set_title("LODA")
# axs[2].set_xlabel("x_1")
# axs[2].set_ylabel("x_2")

plt.savefig(f"Lab3/ex2_3D_hidden_neurons{hidden_neurons}_bins_{bins}.png")