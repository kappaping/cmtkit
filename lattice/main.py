## Main function

from math import *
import numpy as np
import time

import lattice2 as ltc


ltype='sq'
print(ltc.ltcname(ltype))
Nbl=[2,2,1]
Nsl=ltc.slnum(ltype)
bc=1
rs=ltc.ltcsites(Nbl,Nsl)
print(rs)
NB,RD=ltc.ltcpairdist(ltype,rs,Nbl,bc)
print(NB)
nnbs=ltc.nthneighbors(0,NB)
print(nnbs)
nbs=[[NB[nbid0,nbid1] for [nbid0,nbid1] in ltc.nthneighbors(nb,NB)] for nb in range(np.max(NB)+1)]
print(nbs)
