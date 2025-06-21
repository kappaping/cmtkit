## Plot the heat capacity.

import numpy as np
import time
import matplotlib.pyplot as plt
import joblib

import sys
sys.path.append("../lattice/")
import lattice as ltc
sys.path.append("../tightbinding/")
import tightbinding as tb
import monte_carlo




# Lattice setup

# Lattice structure.
ltype = "sq"
Nbl = [16, 16, 1]
rs = ltc.ltcsites(ltype, Nbl)[0]
bc = 1
file_lattice = "../../data/lattice/square/16161_bc_1"
NB = ltc.ltcpairdist(ltype, rs, Nbl, bc, toread=True, filet=file_lattice)[0]

# Ising model Hamiltonian.
Js = [0.,-1. / 2.]
H = tb.tbham(Js, NB, 1).real




data=joblib.load("../../data/monte_carlo/data_ising_16x16")

Ts = []
cTs = []
print(len(data))
data = data[:-1]

# Compute the heat capacity c_T = <E^2> - <E>^2 at all temperatures to observe the phase transition.
for data_point in data:
    T, sample = data_point
    E_sum = 0.
    E_sq_sum = 0.
    for state in sample:
        E = np.sum(H * np.dot(state, state.T)) / state.shape[0]
        E_sum += E
        E_sq_sum += E**2
    Ts.append(T)
    cTs.append(E_sq_sum / len(sample) - (E_sum / len(sample))**2)

plt.plot(Ts,cTs)
plt.xlabel("T")
plt.ylabel("c$_T$")
plt.savefig("../../figs/fig_test.pdf")
plt.show()

