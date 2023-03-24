## Main function

from math import *
import numpy as np
import matplotlib.pyplot as plt

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
nf=5./12.
print('filling = ',nf)
H=lambda k:bdth.tbham(k,htb,ltype,uctype,Nfl)
Nk=50

bzop=False
ks=bdth.brillouinzone(ltype,uctype,Nk,bzop)[0]

todata=True
dataks=[[ks[nk][0],ks[nk][1],0.] for nk in range(len(ks))]

dataks=plbd.fermisurface(H,nf,ltype,uctype,Nk)

filetfig='../../figs/bz.pdf'
tosave=True
plbd.plotbz(ltype,uctype,todata,dataks,tosave,filetfig)




