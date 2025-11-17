from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pyod.models.iforest import IForest
from pyod.models.loda import LODA
from pyod.models.dif import DIF
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import numpy as np


data = loadmat("Lab3/shuttle.mat")

X = data["X"]
y = data["y"].ravel()

y_binary = (y == 1).astype(int) # In this dataset y is 1 when normal, and other classes when an anomaly 
n_splits = 10
ba_results = {"IForest": [], "LODA": [], "DIF": []}
roc_results = {"IForest": [], "LODA": [], "DIF": []}


for i in range(n_splits):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.4, random_state=i, stratify=y_binary
    )


    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)

    models = {
        "IForest": IForest(random_state=i),
        "LODA": LODA(),
        "DIF": DIF()
    }

    for name, model in models.items():
        model.fit(X_train_norm)
        scores = model.decision_function(X_test_norm)
        y_pred = model.predict(X_test_norm)

        ba = balanced_accuracy_score(y_test, y_pred)
        roc = roc_auc_score(y_test, scores)  

        ba_results[name].append(ba)
        roc_results[name].append(roc)
    print(f"Finished {i} iteration")


for name in ["IForest", "LODA", "DIF"]:
    mean_ba = np.mean(ba_results[name])
    mean_roc = np.mean(roc_results[name])
    print(f"{name}: Mean BA = {mean_ba:.4f}, Mean ROC AUC = {mean_roc:.4f}")


# IForest: Mean BA = 0.9762, Mean ROC AUC = 0.9969
# LODA: Mean BA = 0.6538, Mean ROC AUC = 0.6259
# DIF: Mean BA = 0.5182, Mean ROC AUC = 0.9706