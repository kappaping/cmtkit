"""
Algorithm module: Defines the algorithm that optimizes the MERA by energy minimization.
"""


import math
import numpy as np
import sys
sys.path.append("../")
import network
import tensor_tools
import operation




def optimization(ansatz, h, M_ansatz=1e4, M_layer=5, M_one=5, error_max=1e-8, tensor_error_max=1e-8, to_dynamic_M=True):
    """
    Optimization function: Optimize the MERA by an energy minimization with respect to a given Hamiltonian.
    """

    ws = ansatz.ws
    us = ansatz.us
    chi_top = ansatz.chi_top
    rho_top = ansatz.rho_top
    N_layer = ansatz.N_layer

    # Shift the Hamiltonian to guarantee that all of the eigenvalues are negative.
    h_matrix = tensor_tools.get_matrix(h, [[0, 1], [2, 3]])
    es_h = np.linalg.eigvalsh(h_matrix)
    e_shift = -max(abs(es_h)) - 1
    print(f"Shift the energy by {e_shift}.")
    h = h + e_shift * np.identity(len(es_h)).reshape(h.shape)

    e = 0.
    error = 0.1

    # Begin the iterative optimization.
    print("Begin the iterative optimization.")
    for i in range(M_ansatz):
        hs = [h]
        rhos = [rho_top]
        ws_old = ws.copy()
        us_old = us.copy()
        rho_top_old = rho_top.copy()
        e_old = e

        if to_dynamic_M:
            M_layer = 1 + max(0, round(-math.log10(abs(error)) / 2))
            M_one = M_layer
            print(f"    Dynamic iteration = {M_layer}")

        # Create the list of density matrices in all layers by descending from the top density matrix.
        top = True
        for j in range(N_layer - 1):
            rhos.append(operation.descend(ws[-1 - j], us[-1 - j], rhos[j], top))
            top = False
        
        # Update the layers from the bottom to the top.
        e_layer = e
        for j in range(N_layer):
            # Update the isometry and the disentangler in the layer.
            ws[j], us[j], e_layer, error_layer = operation.layer_update(e_layer, ws[j], us[j], hs[j], rhos[-1 - j], M_layer, M_one, error_max)
            print(f"    Layer = {j}, e_layer = {e_layer}, error_layer = {error_layer}")
            if j == N_layer - 1:
                top = True
            # Obtain the Hamiltonian in the next layer by ascending.
            hs.append(operation.ascend(ws[j], us[j], hs[j], top))

        # Obtain the top density matrix and energy.
        rho_top, e = operation.top_ed(hs[-1], chi_top)
        e_print = e - e_shift
        error = e - e_old
        ws_error = max([np.max(np.abs(ws[n_w] - ws_old[n_w])) for n_w in range(len(ws))])
        us_error = max([np.max(np.abs(us[n_u] - us_old[n_u])) for n_u in range(len(us))])
        rho_top_error = np.max(np.abs(rho_top - rho_top_old))
        tensor_error = max([ws_error, us_error, rho_top_error])
        print(f"{i}-th iteration: e = {e_print}, error = {error}, ws_error = {ws_error}, us_error = {us_error}, rho_top_error = {rho_top_error}")
        if abs(error) < error_max and tensor_error < tensor_error_max:
            break

    e = e - e_shift

    ansatz.ws = ws
    ansatz.us = us
    ansatz.rho_top = rho_top

    return e




