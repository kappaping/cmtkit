import numpy as np
import sys
sys.path.append("../")
from tensornetwork import spin_model
from tensornetwork import tensor_tools


h = spin_model.tf_ising(2, 1, 0.3)
h_matrix = tensor_tools.get_matrix(h, [[0, 1], [2, 3]])
print(h_matrix)
