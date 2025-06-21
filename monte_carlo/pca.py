import numpy as np
import joblib
import matplotlib.pyplot as plt

data = joblib.load("../../data/monte_carlo/data_ising_16x16")
data = data[:-1]

T_all = []
sample_all = []
for data_point in data:
    T, sample = data_point
    for state in sample:
        state = state.reshape(-1)
        state = state / np.linalg.norm(state)
        sample_all.append(state)
        T_all.append(T)
sample_all = np.array(sample_all)
U, S, Vh = np.linalg.svd(sample_all)
states_pca = Vh[0:2,:].T

X, Y = np.dot(sample_all, states_pca).T.tolist()

plt.scatter(X, Y, c=T_all, cmap="coolwarm")
plt.savefig("../../figs/fig_test.pdf")
plt.show()
