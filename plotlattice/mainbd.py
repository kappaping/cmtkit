## Main function

from math import *
import numpy as np
import time

import sys
sys.path.append('../lattice')
import lattice as ltc
sys.path.append('../tightbinding')
import tightbinding as tb
import brillouinzone as bz
sys.path.append('../bandtheory')
import bandtheory as bdth
import plotband as plbd


ltype='ka'
print(ltc.ltcname(ltype))
Nbl=[6,6,1]
print('[n0,n1,n2] = ',Nbl)
Nsl=ltc.slnum(ltype)
bc=1
rs=ltc.ltcsites(Nbl,Nsl)
Nr=len(rs)
Nfl=1
Nrfl=[Nr,Nfl]

nf=5./12.

ts=[0.,-1.,-0.]

NB,RD,RDV=ltc.ltcpairdist(ltype,rs,Nbl,True,bc)
H=tb.tbham(ts,NB,Nfl)

prds=[1,1,1]
print('periodicity = ',prds)

Nuc,rucs=bdth.ucsites(ltype,prds)
RUCRP=bdth.ftsites(rs,Nr,prds,Nuc,rucs)

Hk=lambda k:bdth.ftham(k,H,Nrfl,RDV,rucs,RUCRP)
Nk=50

filetfig='../../figs/testbd.pdf'
tosave=True
plbd.plotbandcontour(Hk,ltype,prds,Nfl,Nk,nf,tosave,filetfig)


