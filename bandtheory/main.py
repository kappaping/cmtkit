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
uctype=121
Nfl=1
Nbd=bdth.ucstnum(ltype,uctype,Nfl)
print('Band number = ',Nbd)
htb=[0.,-1.,0.]
mu=0.
H=lambda k:bdth.tbham(htb,k,ltype,uctype,Nfl)
hska=ltc.hskpoints(ltype,uctype)
hsks=[hska[0],hska[1],hska[3],hska[2],hska[0]]
#hsks=[hska[0],hska[1],[hska[5][0],-hska[5][1]],hska[0]]
#hsks=[hska[0],hska[1],[hska[5][0],hska[5][1]],hska[0]]
Nk=50
plbd.plotband(H,mu,Nbd,hsks,Nk)

