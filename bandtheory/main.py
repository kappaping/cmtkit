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
Nbd=bdth.ucstnum(ltype,uctype,Nfl)
print('Band number = ',Nbd)
htb=[0.,-1.,0.]
H=lambda k:bdth.tbham(htb,k,ltype,uctype,Nfl)
hsksa=ltc.hskpoints(ltype)
hsks=[hsksa[0],hsksa[1],-1.*hsksa[5],-1.*hsksa[3],hsksa[0]]
Nk=10
plbd.plotband(H,Nbd,hsks,Nk)

