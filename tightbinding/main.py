## Main function

from math import *
import numpy as np
import time

import sys
sys.path.append('../lattice')
import lattice as ltc
import tightbinding as tb


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
ltcss=ltc.ltcsites(Nall[0])

[print([ltcs,fl],tb.stid(ltcs,fl,Nall)) for ltcs in ltcss for fl in range(Nfl)]

htb=[0.,-1.,0.]
Mt=np.zeros((Nst,Nst),dtype=complex)
time1=time.time()
tb.tbham(Mt,htb,ltcss,Nall,bc,ltype)
time2=time.time()
print('time = ',time2-time1)
print(Mt)



