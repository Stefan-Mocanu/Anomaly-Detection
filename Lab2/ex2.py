from pyod.utils.data import generate_data_clusters
from pyod.models import knn
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

X_train, X_test, Y_train, Y_test = generate_data_clusters(400,200, random_state=31, n_clusters=2, n_features=2)
neighbors = 5

# For 10 neighbors: Balanced accuracy: 0.9914285714285714
# For 5 neighbors: Balanced accuracy:  0.9885714285714285
# For 2 neighbors: Balanced accuracy:  0.9742857142857142

model = knn.KNN(n_neighbors=neighbors)

model.fit(X_train)

Y_train_pred = model.predict(X_train)
Y_test_pred = model.predict(X_test)

balanced_acc = metrics.balanced_accuracy_score(Y_test, Y_test_pred)

print("Balanced accuracy: ", balanced_acc)

fig,axs = plt.subplots(nrows=2, ncols=2)



axs[0,0].scatter(X_train[Y_train == 0, 0], X_train[Y_train == 0, 1], c='blue', label='Inliers', alpha=0.6)
axs[0,0].scatter(X_train[Y_train == 1, 0], X_train[Y_train == 1, 1], c='red', label='Outliers', alpha=0.8)
axs[0,0].set_title('Ground truth training')
axs[0,0].legend()

axs[0,1].scatter(X_train[Y_train_pred == 0, 0], X_train[Y_train_pred == 0, 1], c='blue', label='Inliers', alpha=0.6)
axs[0,1].scatter(X_train[Y_train == 1, 0], X_train[Y_train == 1, 1], c='red', label='Outliers', alpha=0.8)
axs[0,1].set_title('Predicted truth training')
# axs[0,1].legend()

axs[1,0].scatter(X_test[Y_test == 0, 0], X_test[Y_test == 0, 1], c='blue', label='Inliers', alpha=0.6)
axs[1,0].scatter(X_test[Y_test == 1, 0], X_test[Y_test == 1, 1], c='red', label='Outliers', alpha=0.8)
axs[1,0].set_title('Ground truth testing')
# axs[1,0].legend()

axs[1,1].scatter(X_test[Y_test_pred == 0, 0], X_test[Y_test_pred == 0, 1], c='blue', label='Inliers', alpha=0.6)
axs[1,1].scatter(X_test[Y_test == 1, 0], X_test[Y_test == 1, 1], c='red', label='Outliers', alpha=0.8)
axs[1,1].set_title('Predicted truth testing')
# axs[1,1].legend()

plt.savefig("Lab2/ex2.png")