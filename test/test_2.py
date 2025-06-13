import numpy as np
import sys
sys.path.append("../")
from tensornetwork import network

diagram_name = "../tensornetwork/mera/diagrams/Env_u_R.dgm"
w = np.zeros((8, 2, 2, 2))
u = np.zeros((2, 2, 2, 2))
h = np.zeros((2, 2, 2, 2))
rho = np.zeros((8, 8, 8, 8))
tensor_list = [("w0", w), ("w1", w), ("w0d", w), ("w1d", w), ("ud", u), ("h", h), ("rho", rho)]
tensors = {tensor[0]:tensor[1] for tensor in tensor_list}

network_0 = network.network(diagram_name, tensors)
print(network_0)
print(network_0.diagram)
print(network_0.order)
print(network_0.ext_bonds)

print("Tensor network contraction =", network_0.contract(to_print=True).shape)


