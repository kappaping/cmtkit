"""
Operation module: Defines the operations that are used in the MERA algorithm.
"""


import numpy as np
import sys
sys.path.append("../")
import network
import tensor_tools




def ascend(w, u, o, top):
    """
    Ascend function: Ascend a two-site observable by one layer.
    """

    diagram_names = ["diagrams/Ascend_L.dgm", "diagrams/Ascend_C.dgm", "diagrams/Ascend_R.dgm"]
    wd, ud = w.conj(), u.conj()
    o_0 = o
    tensors = {"w0": w, "w1": w, "w0d": wd, "w1d": wd, "u": u, "ud": ud, "o": o_0}
    nets = [network.network(diagram_name, tensors) for diagram_name in diagram_names]
    o_1 = sum([net.contract() for net in nets])/3.

    if top:
        o_1_inversion = np.moveaxis(o_1, [1, 0, 3, 2], [0, 1, 2, 3])
        o_1 = (o_1 + o_1_inversion)/2.

    return o_1


def descend(w, u, rho, top):
    """
    Descend function: Descend a two-site density matrix by one layer.
    """
    
    rho_1 = rho
    if top:
        rho_1_inversion = np.moveaxis(rho_1, [1, 0, 3, 2], [0, 1, 2, 3])
        rho_1 = (rho_1 + rho_1_inversion)/2.

    diagram_names = ["diagrams/Descend_L.dgm", "diagrams/Descend_C.dgm", "diagrams/Descend_R.dgm"]
    wd, ud = w.conj(), u.conj()
    tensors = {"w0": w, "w1": w, "w0d": wd, "w1d": wd, "u": u, "ud": ud, "rho": rho_1}
    nets = [network.network(diagram_name, tensors) for diagram_name in diagram_names]
    rho_0 = sum([net.contract() for net in nets])/3.

    return rho_0


def environment_w(w, u, h, rho):
    """
    Environment-w function: Create the environment of an isometry w.
    """
    
    diagram_names = ["diagrams/Env_w_LL.dgm", "diagrams/Env_w_LC.dgm", "diagrams/Env_w_LR.dgm", "diagrams/Env_w_RL.dgm", "diagrams/Env_w_RC.dgm", "diagrams/Env_w_RR.dgm"]
    wd, ud = w.conj(), u.conj()
    tensors = {"w1": w, "w0d": wd, "w1d": wd, "u": u, "ud": ud, "h": h, "rho": rho}
    nets = [network.network(diagram_name, tensors) for diagram_name in diagram_names]
    Y_w = sum([net.contract() for net in nets])/6.

    return Y_w


def environment_u(w, u, h, rho):
    """
    Environment-u function: Create the environment of a disentangler u.
    """
    
    diagram_names = ["diagrams/Env_u_L.dgm", "diagrams/Env_u_C.dgm", "diagrams/Env_u_R.dgm"]
    wd, ud = w.conj(), u.conj()
    tensors = {"w0": w, "w1": w, "w0d": wd, "w1d": wd, "ud": ud, "h": h, "rho": rho}
    nets = [network.network(diagram_name, tensors) for diagram_name in diagram_names]
    Y_u = sum([net.contract() for net in nets])/3.

    return Y_u


def minimize(w, u, h, rho, tensor_type):
    """
    Minimization function: Obtain the optimized isometry or disentangler from the environment that minimizes the energy.
    """

    if tensor_type == "w":
        Y = environment_w(w, u, h, rho)
        bonds_io = [[1, 2, 3], [0]]
    elif tensor_type == "u":
        Y = environment_u(w, u, h, rho)
        bonds_io = [[2, 3], [0, 1]]
    Y_matrix = tensor_tools.get_matrix(Y, bonds_io)

    U, S, Vh = np.linalg.svd(Y_matrix)
    dim_S = S.shape[0]
    tensor = -np.dot(Vh[:dim_S, :].T.conj(), U[:, :dim_S].T.conj())
    if tensor_type == "w":
        tensor = np.reshape(tensor, w.shape)
    elif tensor_type == "u":
        tensor = np.reshape(tensor, u.shape)
    
    e = -sum(S)

    return tensor, e


def update_tensor(e, w, u, h, rho, tensor_type, M_one, error_max):
    """
    Update tensor function: Update an isometry or disentangler iteratively under energy minimization.
    """
    e_old = e
    error = 1.
    for i in range(M_one):
        tensor, e = minimize(w, u, h, rho, tensor_type)
        if tensor_type == "w":
            w = tensor
        elif tensor_type == "u":
            u = tensor
        error = e - e_old
        e_old = e
        if abs(error) < error_max:
            break
    print(f"        Tensor type: {tensor_type}, error = {error}")

    return tensor, e


def layer_update(e, w, u, h, rho, M_layer, M_one, error_max):
    """
    Layer update function: Update the layer under energy minimization.
    """
    e_old = e
    tensor_wu = [w, u]
    tensor_types = ["w", "u"]
    for i in range(M_layer):
        for j in range(2):
            tensor_wu[j], e = update_tensor(e, tensor_wu[0], tensor_wu[1], h, rho, tensor_types[j], M_one, error_max)
        error = e - e_old
        if abs(error) < error_max:
            break
    w, u = tensor_wu

    return w, u, e, error


def top_ed(h, chi_top):
    """
    Top ED function: Do the exact diagonalization at the top layer to get the top density matrix.
    """
    h_matrix = tensor_tools.get_matrix(h, [[0, 1], [2, 3]])
    es, U = np.linalg.eigh(h_matrix)
    e = sum(es[:chi_top])
    dim_bond = h.shape[0]
    U = U[:, :chi_top].reshape(dim_bond, dim_bond, chi_top)
    Ud = U.conj()
    rho_top = np.tensordot(U, Ud, (2, 2))

    return rho_top, e








