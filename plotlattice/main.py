## Main function

from math import *
import numpy as np
import time

import sys
sys.path.append('../lattice')
import lattice as ltc
sys.path.append('../tightbinding')
import tightbinding as tb
import plotlattice as pltc


ltype='ka'
print(ltc.ltcname(ltype))
Nbl=[2,2,1]
Nsl=ltc.slnum(ltype)
Nltc=[Nbl,Nsl]
Nfl=2
Nall=[Nltc,Nfl]
Nst=tb.stnum(Nall)
print('State number = ',Nst)
bc=1
rs=ltc.ltcsites(Nall[0])

pltc.plotlattice(rs,Nall,ltype)


