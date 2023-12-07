from math import *
import numpy as np
import bogoliubovdegennes as bdg

A=np.array([[1,2],[3,4]])
print(A)
B=bdg.phmattobdg(A)
print(B)
BBs=[[bdg.bdgblock(B,phid0,phid1) for phid1 in range(2)] for phid0 in range(2)]
print(BBs)
