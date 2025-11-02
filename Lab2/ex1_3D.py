import numpy as np
import matplotlib.pyplot as plt

a = 2
b = 3
c = 2.7
SIZE = 1000

mu = 4
var = 0.3
big_var = 4

def leverage_scores(x):
    X = np.vstack((np.ones(SIZE), x)).T
    U, _, _ = np.linalg.svd(X, full_matrices=False)
    h = np.sum(U**2, axis=1)
    return h


regular_x1 = np.random.normal(loc=mu, scale=var, size=SIZE)
spread_x1 = np.random.normal(loc=mu, scale=big_var, size=SIZE)
regular_x2 = np.random.normal(loc=mu, scale=var, size=SIZE)
spread_x2 = np.random.normal(loc=mu, scale=big_var, size=SIZE)
regular_epsilon = np.random.normal(loc=mu, scale=var, size=SIZE)
spread_epsilon = np.random.normal(loc=mu, scale=big_var, size=SIZE)



fig, axs = plt.subplots(
    nrows=4, ncols=2,
    figsize=(14, 18),               # wider and taller for 8 subplots
    subplot_kw={'projection': '3d'}
)
axs.flatten()
places = [[0,0],[0,1],[1,0],[1,1],[2,0],[2,1],[3,0],[3,1]]
index = 0
for x1,x1_type in [(regular_x1,"regular"),(spread_x1,"spread")]:
    for x2,x2_type in [(regular_x2,"regular"),(spread_x2,"spread")]:
        for epsilon,epsilon_type in [(regular_epsilon,"regular"),(spread_epsilon,"spread")]:
            y = a*x1 + b*x2 + c + epsilon

            print(np.isnan(x1).any(), np.isnan(x2).any(), np.isnan(y).any())
            print(np.isinf(x1).any(), np.isinf(x2).any(), np.isinf(y).any())

            h = leverage_scores(np.vstack((x1,x2)))
            high_idx = np.argsort(h)[-30:]
            i, j = places[index]
            axs[i,j].scatter(x1,x2,y, label='Data')
            axs[i,j].scatter(x1[high_idx],
                               x2[high_idx],
                               y[high_idx],
                               color='red',
                               label='High Leverage')
            axs[i,j].set_title(f'{x1_type} x1, {x2_type} x2, {epsilon_type} y')
            # axs[i,j].legend()
            index+=1

plt.tight_layout()
plt.savefig("Lab2/ex1_3D.png")

