import numpy as np
import matplotlib.pyplot as plt

a = 2
b = 3
SIZE = 1000

mu = 4
var = 0.7
big_var = 2

def leverage_scores(x):
    X = np.vstack((np.ones(SIZE), x)).T
    U, _, _ = np.linalg.svd(X, full_matrices=False)
    h = np.sum(U**2, axis=1)
    return h


regular_x = np.random.normal(loc=mu, scale=var, size=SIZE)
spread_x = np.random.normal(loc=mu, scale=big_var, size=SIZE)
regular_epsilon = np.random.normal(loc=mu, scale=var, size=SIZE)
spread_epsilon = np.random.normal(loc=mu, scale=big_var, size=SIZE)



regular_y_regular_x = a*regular_x + b + regular_epsilon
regular_y_spread_x = a*spread_x + b + regular_epsilon
spread_y_regular_x = a*regular_x + b + spread_epsilon
spread_y_spread_x = a*spread_x + b + spread_epsilon

regular_h = leverage_scores(regular_x)
spread_h = leverage_scores(spread_x)
high_idx_regular = np.argsort(regular_h)[-30:]
high_idx_spread = np.argsort(spread_h)[-30:]


fig,axs = plt.subplots(nrows=2, ncols=2)
axs[0, 0].scatter(regular_x,regular_y_regular_x, label='Data')
axs[0, 0].scatter(regular_x[high_idx_regular]
                  , regular_y_regular_x[high_idx_regular]
                  , color='red', label='High leverage')
axs[0, 0].set_title('Regular X, Regular Y')
axs[0, 0].legend()

axs[0, 1].scatter(spread_x,regular_y_spread_x, label='Data')
axs[0, 1].scatter(spread_x[high_idx_spread]
                  , regular_y_spread_x[high_idx_spread]
                  , color='red', label='High leverage')
axs[0, 1].set_title('Spread X, Regular Y')
axs[0, 1].legend()

axs[1, 0].scatter(regular_x,spread_y_regular_x, label='Data')
axs[1, 0].scatter(regular_x[high_idx_regular]
                  , spread_y_regular_x[high_idx_regular]
                  , color='red', label='High leverage')
axs[1, 0].set_title('Regular X, Spread Y')
axs[1, 0].legend()

axs[1, 1].scatter(spread_x,spread_y_spread_x, label='Data')
axs[1, 1].scatter(spread_x[high_idx_spread]
                  , spread_y_spread_x[high_idx_spread]
                  , color='red', label='High leverage')
axs[1, 1].set_title('Spread X, Spread Y')
axs[1, 1].legend()


plt.savefig("Lab2/ex1_2D.png")