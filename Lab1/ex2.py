import pyod.models.knn as knn
from pyod.utils.data import generate_data 
import sklearn.metrics as metrics
import numpy as np
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = generate_data(400,100,2,0.1)

model = knn.KNN(contamination=0.01)
# daca contaminarea este mai mare decat contaminarea din dataset atunci creste nr de FP
# daca contaminarea este mai mica decat cont din dataset atunci creste nr de FN
model.fit(X_train)

train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)
train_scores = model.decision_function(X_train)
test_scores = model.decision_function(X_test)

true_y = np.concatenate([y_train, y_test])
predictions = np.concatenate([train_predictions,test_predictions])
scores = np.concatenate([train_scores, test_scores])

confusion = metrics.confusion_matrix(true_y,predictions)

print("Confusion matrix:\n", confusion)

balanced_acc = metrics.balanced_accuracy_score(true_y, predictions)

print("Balanced Accuracy:", balanced_acc)

fpr, tpr, thresholds = metrics.roc_curve(true_y,scores)
roc_auc = roc_auc_score = metrics.roc_auc_score(true_y, scores)

plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
plt.savefig("Lab1/ex2.png")


