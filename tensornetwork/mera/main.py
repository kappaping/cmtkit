"""
Main function: Define the Hamiltonian and get the MERA for the ground state.
"""


import numpy as np
import sys
sys.path.append("../")
import spin_model
import mera




N_layer = 5
d = 2
chi = 8
ansatz = mera.mera(N_layer, d, chi)

J = 1.
hx = 1.
h = spin_model.tf_ising(J, hx)
#h = spin_model.heisenberg(J)

M_ansatz = 10000
M_layer = 2
M_one = 2

e = ansatz.optimize(h, M_ansatz, M_layer, M_one)


