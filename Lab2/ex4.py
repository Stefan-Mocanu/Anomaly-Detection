from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pyod.models.knn import KNN
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from pyod.utils.utility import standardizer
from pyod.models.combination import average, maximization

data = loadmat('Lab2/cardio.mat')

X = data['X']
y = data['y'].ravel() 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

neighbors_range = np.linspace(30, 120, 10, dtype=int)


train_scores = []
test_scores = []

contamination = 0.09

for n in neighbors_range:
    model = KNN(n_neighbors=n, contamination=contamination)
    model.fit(X_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    ba_train = balanced_accuracy_score(y_train, y_train_pred)
    ba_test = balanced_accuracy_score(y_test, y_test_pred)

    print(f"n_neighbors={n}:")
    print(f"\tTrain BA={ba_train:.3f} ")
    print(f"\tTest BA={ba_test:.3f}")

    train_scores.append(model.decision_scores_)
    test_scores.append(model.decision_function(X_test))

train_scores_norm, test_scores_norm = standardizer(
    np.array(train_scores).T, np.array(test_scores).T
)
    
combined_scores_avg = average(test_scores_norm)
combined_scores_max = maximization(test_scores_norm)



threshold_avg = np.quantile(combined_scores_avg, 1 - contamination)
threshold_max = np.quantile(combined_scores_max, 1 - contamination)

# Predict (1 = outlier, 0 = inlier)
y_pred_avg = (combined_scores_avg > threshold_avg).astype(int)
y_pred_max = (combined_scores_max > threshold_max).astype(int)

ba_avg = balanced_accuracy_score(y_test, y_pred_avg)
ba_max = balanced_accuracy_score(y_test, y_pred_max)

print(f"Balanced Accuracy (Average rule): {ba_avg:.3f}")
print(f"Balanced Accuracy (Maximization rule): {ba_max:.3f}")

# Output:
# n_neighbors=30:
#         Train BA=0.666 
#         Test BA=0.722
# n_neighbors=40:
#         Train BA=0.687 
#         Test BA=0.746
# n_neighbors=50:
#         Train BA=0.706 
#         Test BA=0.758
# n_neighbors=60:
#         Train BA=0.714 
#         Test BA=0.760
# n_neighbors=70:
#         Train BA=0.714 
#         Test BA=0.769
# n_neighbors=80:
#         Train BA=0.715 
#         Test BA=0.769
# n_neighbors=90:
#         Train BA=0.726 
#         Test BA=0.769
# n_neighbors=100:
#         Train BA=0.727 
#         Test BA=0.782
# n_neighbors=110:
#         Train BA=0.735 
#         Test BA=0.783
# n_neighbors=120:
#         Train BA=0.742 
#         Test BA=0.782
# Balanced Accuracy (Average rule): 0.758
# Balanced Accuracy (Maximization rule): 0.758