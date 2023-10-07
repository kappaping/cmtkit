## Main function

from math import *
import numpy as np
import time
import joblib

import lattice2 as ltc


ltype='py'
print(ltc.ltcname(ltype))
Nbl=[6,6,6]
Nsl=ltc.slnum(ltype)
bc=1
rs=ltc.ltcsites(Nbl,Nsl)

filet='../../data/lattice/pyrochlore/666_bc_1'

NB,RD=ltc.ltcpairdist(ltype,rs,Nbl,bc)

joblib.dump([NB,RD],filet)

#[NB,RD]=joblib.load(filet)
#print(NB)
#print(RD)
