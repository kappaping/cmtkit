## Main function

from math import *
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../lattice')
import lattice2 as ltc
import brillouinzone as bz
sys.path.append('../bandtheory')
import bandtheory2 as bdth
import plotband2 as plbd


ltype='tr'
print(ltc.ltcname(ltype))

prds=[1,1,1]

Nk=12

bzop=True
ks=bz.listbz(ltype,prds,Nk,bzop)[0]

todata=True
dataks=[[ks[nk][0],ks[nk][1],0.] for nk in range(len(ks))]

#dataks=plbd.fermisurface(H,nf,ltype,uctype,Nk)

filetfig='../../figs/bz.pdf'
tosave=True
tolabel=False
plbd.plotbz(ltype,prds,todata,dataks,tolabel,tosave,filetfig)




