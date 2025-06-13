"""
MERA module: Defines the class of multiscale entanglement renormalization ansatz (MERA) tensor network.
"""


import numpy as np
from scipy.stats import ortho_group, unitary_group
import joblib
import algorithm




class mera():
    # A class of MERA tensor network.
    # ----------
    # Attributes:
    # num_layer: Number of layers.
    # dim_phys: Physical dimension of the lattice.
    # dim_bond: Maximal bond dimension.
    # dim_top: Number of the states to consider in the top density matrix.
    # isoms: A list of isometries, from bottom to top.
    # disents: A list of disentanglers, from bottom to top.
    # den_mat_top: Top density matrix.
    # ----------


    def __init__(self, N_layer, d, chi, chi_top=1, ws=[], us=[], rho_top=np.array([]), to_randomize=False, to_read=False, file_name="", set_real=True):
        self.num_layer = N_layer
        self.dim_phys = d
        self.dim_bond = chi
        self.dim_top = chi_top
        self.set_ansatz(N_layer, d, chi, chi_top, ws, us, rho_top, to_randomize=to_reandomize, to_read=to_read, file_name=file_name, set_real=set_real)


    def set_ansatz(self, N_layer, d, chi, chi_top=1, ws=[], us=[], rho_top=[], to_randomize=False, to_read=False, file_name="", set_real=True):
        """
        Set up the tensors in MERA from the given tensors.
        """
        if to_read:
            self.read(file_name)
        elif min(len(ws), len(us)) == 0 or to_randomize:
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

        self.isoms = ws
        self.disents = us
        self.den_mat_top = rho_top


    def optimize(self, h, M_ansatz=1e4, M_layer=5, M_one=5, error_max=1e-8, tensor_error_max=1e-8, to_dynamic_M=True):
        """
        Call optimization() in the algorithm.py module to optimize the MERA by energy minimization.
        """
        e = algorithm.optimization(self, h, M_ansatz=M_ansatz, M_layer=M_layer, M_one=M_one, error_max=error_max, tensor_error_max=tensor_error_max, to_dynamic_M=to_dynamic_M)

        return e


    def save(self, file_name):
        """
        Save the MERA to a file.
        """
        joblib.dump(self, file_name)


    def read(self, file_name):
        """
        Read the MERA from a file.
        """
        mera_read = joblib.load(file_name)
        # Clear the existing attributes in self.
        for attr in list(self.__dict__.keys()):
            delattr(self, attr)
        # Assign the attributes of the read MERA to self.
        for attr_name, attr_value in mera_read.__dict__.items():
            setattr(self, attr_name, attr_value)
        print("Read the MERA from:", file_name)




