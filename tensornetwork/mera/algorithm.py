"""
Algorithm module: Defines the algorithm that optimizes the MERA by energy minimization.
"""


import numpy as np
import sys
sys.path.append("../")
import network
import tensor_tools
import operation




def optimization(ansatz, h, M_ansatz, M_layer, M_one, error_max=1e-8):
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
    e_old = e
    rho_top_old = rho_top

    # Begin the iterative optimization.
    print("Begin the iterative optimization.")
    for i in range(M_ansatz):
        hs = [h]
        rhos = [rho_top]
        top = True

        # Create the list of density matrices in all layers by descending from the top density matrix.
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
        e_old = e
        rho_top_error = np.max(np.abs(rho_top - rho_top_old))
        rho_top_old = rho_top
        print(f"{i}-th iteration: e = {e_print}, error = {error}, rho_top_error = {rho_top_error}")
        if abs(error) < error_max:
            break

    e = e - e_shift

    ansatz.ws = ws
    ansatz.us = us
    ansatz.rho_top = rho_top

    return e




