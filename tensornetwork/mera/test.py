import mera

N_layer = 4
d = 2
chi = 8
mera_0 = mera.mera(N_layer, d, chi, to_randomize=True)
print(mera)
print("Number of layers = ", mera_0.N_layer)
print("Site dimension = ", mera_0.d)
print("Bond dimension = ", mera_0.chi)
print("Top state dimension = ", mera_0.chi_top)
ws = mera_0.ws
print("w shapes:", [w.shape for w in ws])
us = mera_0.us
print("u shapes:", [u.shape for u in us])
rho_top = mera_0.rho_top
print("rho_top shapes:", rho_top.shape)
