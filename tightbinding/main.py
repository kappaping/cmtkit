## Main function

from math import *
import numpy as np

import sys
sys.path.append('../common')
import lattice as ltc
import tightbinding as tb


mtype=22
print(ltc.ltcname(mtype))
Nbl=[2,2,1]
Nsl=ltc.nslf(mtype)
Nltc=[Nbl,Nsl]
Nfl=2
Nall=[Nltc,Nfl]
Ndof=Nall[0][0][0]*Nall[0][0][1]*Nall[0][0][2]*Nall[0][1]*Nall[1]
bc=1
ltcss=ltc.ltcsites(Nall[0])

[print([ltcs,fl],tb.stid(ltcs,fl,Nall)) for ltcs in ltcss for fl in range(Nfl)]

htb=[0.,-1.,0.]
Mt=np.zeros((Ndof,Ndof))
tb.tbham(Mt,htb,ltcss,Nall,bc,mtype)
print(Mt)



