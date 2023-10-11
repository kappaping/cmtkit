# Create the data.
from math import *
import numpy as np

A=np.array([[n,n+1] for n in range(5)])
print(A)
B=A.reshape(-1,1,2)
print(B)
C=np.concatenate([B[:-1],B[1:]],axis=1)
print(C)
