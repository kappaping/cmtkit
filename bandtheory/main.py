## Main function

from math import *
import numpy as np
import time

import sys
sys.path.append('../lattice')
import lattice as ltc
import bandtheory as bdth
import plotband as plbd


ltype='ka'
print(ltc.ltcname(ltype))
uctype=111
Nfl=1
htb=[0.,-1.,0.]
nf=2./3.
print('filling = ',nf)
mu=0.
H=lambda k:bdth.tbham(k,htb,ltype,uctype,Nfl)
Nk=50

filetfig='../../figs/kagomeband.pdf'
tosave=True
plbd.plotbandcontour(H,mu,ltype,uctype,Nfl,Nk,nf,tosave,filetfig)

