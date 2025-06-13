import math
import numpy as np
import sys
sys.path.append("../")
from tensornetwork import spin_model
from tensornetwork import tensor_tools


state = np.array([0., 1. / math.sqrt(2), -1. / math.sqrt(2), 0.])
print("state = ", state)

u = np.array([[0., 1./math.sqrt(2), -1./ math.sqrt(2), 0.], [0., 1./math.sqrt(2), 1./ math.sqrt(2), 0.], [1., 0., 0., 0.], [0., 0., 0., 1.]])
print("u = ", u)

print("u.ud = ", np.dot(u, u.T.conj()))

print("u.state =", np.dot(u, state))

u = u.reshape((2, 2, 2, 2))
u = np.moveaxis(u, [0, 2, 1, 3], [0, 1, 2, 3]).reshape(4, 4)
print(np.linalg.svd(u)[1])
