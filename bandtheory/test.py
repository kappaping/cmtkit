from math import *
import numpy as np

import bandtheory2 as bdth

'''
ltype='ka'
periods=[23,23,1]
rs=bdth.ucsites(ltype,periods)
print(rs)
'''

A=np.array([[1,2],[3,4],[5,6]])
print([A[0],A[1]])
print(np.argwhere(A==np.array([1,2])))
