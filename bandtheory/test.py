from math import *
import numpy as np


x=np.array([np.array([[1,0],[0,-1]]),np.array([[2,0],[0,-2]])])
print(np.linalg.eigvalsh(x))
