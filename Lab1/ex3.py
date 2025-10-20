from pyod.utils.data import generate_data 
from sklearn.metrics import balanced_accuracy_score
import numpy as np

X_train, X_test, y_train, y_test = generate_data(1000,0,1,0.1)

mean = np.mean(X_train)
std = np.std(X_train)

z_scores = (X_train-mean)/std

threshold = np.quantile(np.abs(z_scores), 1 - 0.1)

y_pred = (np.abs(z_scores) > threshold).astype(int)

balanced_acc = balanced_accuracy_score(y_train, y_pred)

print("Balanced accuracy:", balanced_acc)