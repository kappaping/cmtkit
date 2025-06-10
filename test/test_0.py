import numpy as np
import sys
sys.path.append("../")
from tensornetwork import network

"""
diagram_name = "diagram_test_0.dgm"
tensor_list = [("A", np.array([[0, 1], [2, 3]])), ("B", np.array([[4, 5], [6, 7]])), ("C", np.array([[8, 9], [10, 11]]))]
tensors = {tensor[0]:tensor[1] for tensor in tensor_list}

network_0 = network.network(diagram_name, tensors)
print(network_0)
print(network_0.diagram)
print(network_0.order)
print(network_0.ext_bonds)
print(network_0.tensors)

print("Tensor network contraction =", network_0.contract(toprint=True))

print("Multidot =", np.linalg.multi_dot([tensor_list[0][1], tensor_list[1][1], tensor_list[2][1]]))
"""
#"""
diagram_name = "diagram_test_2.dgm"
tensor_list = [("A", np.array([[[0, 1], [2, 3]],[[4, 5], [6, 7]]]))]
tensors = {tensor[0]:tensor[1] for tensor in tensor_list}

network_0 = network.network(diagram_name, tensors)
print(network_0)
print(network_0.diagram)
print(network_0.order)
print(network_0.ext_bonds)
print(network_0.tensors)

print("Tensor network contraction =", network_0.contract(toprint=True))

print("trace =", np.trace(tensor_list[0][1], axis1=0, axis2=2))
#"""

