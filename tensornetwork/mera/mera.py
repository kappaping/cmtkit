"""
MERA module: Defines the class of multiscale entanglement renormalization ansatz (MERA) tensor network.
"""


import numpy as np
from scipy.stats import ortho_group, unitary_group
import algorithm




class mera():
    # 


    def __init__(self, N_layer, d, chi, chi_top=1, ws=[], us=[], rho_top=np.array([]), to_randomize=False, set_real=True):
        self.N_layer = N_layer
        self.d = d
        self.chi = chi
        self.chi_top = chi_top
        self.set_ansatz(N_layer, d, chi, chi_top, ws, us, rho_top, to_randomize, set_real)


    def set_ansatz(self, N_layer, d, chi, chi_top=1, ws=[], us=[], rho_top=[], to_randomize=False, set_real=True):
        """
        Set up the tensors in MERA from the given tensors.
        """

        if min(len(ws), len(us)) == 0 or to_randomize:
            print("Initialize a MERA by randomization.")
            dim_0 = d
            if set_real:
                print("Set the MERA as real: Randomize from an orthogonal random vector set.")
                f_rvs = ortho_group.rvs
            else:
                print("Set the MERA as complex: Randomize from a unitary random vector set.")
                f_rvs = unitary_group.rvs
            for n in range(N_layer):
                dim_1 = min(dim_0**3, chi)
                w = f_rvs(dim_0**3)[:dim_1, :].reshape((dim_1, dim_0, dim_0, dim_0))
                u = f_rvs(dim_0**2).reshape((dim_0, dim_0, dim_0, dim_0))
                ws.append(w)
                us.append(u)
                dim_0 = dim_1
            U = f_rvs(dim_0**2)[:, :chi_top].reshape(dim_0, dim_0, chi_top)
            Ud = U.conj()
            rho_top = np.tensordot(U, Ud, (2, 2))

        self.ws = ws
        self.us = us
        self.rho_top = rho_top


    def optimize(self, h, M_ansatz=1e4, M_layer=5, M_one=5, error_max=1e-8, tensor_error_max=1e-8, to_dynamic_M=True):
        e = algorithm.optimization(self, h, M_ansatz=M_ansatz, M_layer=M_layer, M_one=M_one, error_max=error_max, tensor_error_max=tensor_error_max, to_dynamic_M=to_dynamic_M)

        return e




