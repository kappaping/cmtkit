## Main function

from math import *
import numpy as np
import time
import joblib

import lattice as ltc


ltype='tr'
print(ltc.ltcname(ltype))
Nbl=[24,24,1]
Nsl=ltc.slnum(ltype)
bc=1
rs=ltc.ltcsites(Nbl,Nsl)

filet='../../data/lattice/triangular/24241_bc_1'

NB,RD,RDV=ltc.ltcpairdist(ltype,rs,Nbl,bc,tordv=True)
print(NB)
print(RD)
print(RDV)
print('rdv0 = ',[RDV[0,n] for n in range(np.shape(RDV)[0])])

joblib.dump([NB,RD],filet)

#[NB,RD]=joblib.load(filet)
#print(NB)
#print(RD)
