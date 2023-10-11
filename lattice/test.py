## This is a test

from math import *
import numpy as np
import lattice2 as ltc
import brillouinzone as bz

ltype='tr'
prds=[1,1,1]

#print(bz.ucblvecs(ltype,prds))
#print(bz.hskpoints(ltype,prds))
a=np.array([True,True,True])
b=np.array([1,1,1])
print('a=b: ',np.array_equal(a,b))
