"""
Tensor tools: A module for the convenient tools when working with tensors.
"""


import math
import numpy as np




def get_matrix(tensor, bonds_io):
    """
    Get matrix function: Get the matrix form of a tensor, with the in and out bonds given by bonds_io.
    """

    tensor_temp = np.moveaxis(tensor, bonds_io[0] + bonds_io[1], range(tensor.ndim))
    dims = tensor_temp.shape
    dim_in, dim_out = math.prod(dims[:len(bonds_io[0])]), math.prod(dims[len(bonds_io[0]):])
    matrix = tensor_temp.reshape((dim_in, dim_out))

    return matrix




