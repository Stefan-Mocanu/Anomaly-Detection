from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

X, _= make_blobs(500,2,centers = 1, cluster_std=1.0, center_box=(0.0,0.0))

p_vectors = np.random.multivariate_normal(mean=(0,0),cov=[[1,0],[0,1]], size=5)

p_vectors /= np.linalg.norm(p_vectors, axis=1,keepdims=True)

hists = []
probabilities = []

number_of_bins = 100

for v in p_vectors:
    proj = X @ v
    hist, bin_edges = np.histogram(proj, bins=number_of_bins, range=(-5,5))
    prob = hist/hist.sum()
    hists.append((hist,bin_edges))
    probabilities.append(prob)

scores = []

for x in X:
    sample_probs = []
    for v, (hist, bin_edges), probs in zip(p_vectors, hists, probabilities):
        proj = x @ v
        bin_idx = np.digitize(proj, bin_edges) - 1
        if 0 <= bin_idx < len(probs):
            sample_probs.append(probs[bin_idx])
        else:
            sample_probs.append(0)
    scores.append(np.mean(sample_probs))

X_test = np.random.uniform(low=-3,high=3,size=(500,2))

scores_test = []
for x in X_test:
    sample_probs = []
    for v, (hist, bin_edges), probs in zip(p_vectors, hists, probabilities):
        proj = x @ v
        bin_idx = np.digitize(proj, bin_edges) - 1
        if 0 <= bin_idx < len(probs):
            sample_probs.append(probs[bin_idx])
        else:
            sample_probs.append(0)
    scores_test.append(np.mean(sample_probs))


plt.figure(figsize=(6,6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=scores_test, cmap='viridis')
plt.colorbar(label="Anomaly score")
plt.title("Test dataset with anomaly score")
plt.xlabel("x_1")
plt.ylabel("x_2")
plt.savefig(f"Lab3/ex1_{number_of_bins}_bins.png")

