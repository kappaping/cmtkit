import math
import mera

N_layer = 4
d = 2
chi = 8
mera_0 = mera.mera(N_layer, d, chi, to_randomize=True)
print(mera)
print("Number of layers = ", mera_0.num_layer)
print("Site dimension = ", mera_0.dim_phys)
print("Bond dimension = ", mera_0.dim_bond)
print("Top state dimension = ", mera_0.dim_top)
ws = mera_0.isoms
print("w shapes:", [w.shape for w in ws])
us = mera_0.disents
print("u shapes:", [u.shape for u in us])
rho_top = mera_0.den_mat_top
print("rho_top shapes:", rho_top.shape)

A = [0, 1, 2, 3]
print("before: A =", A)

def append1(A):
    B = A + [1]
    A = B
    return B

B = append1(A)
print("after:")
print("A =", A)
print("B =", B)
