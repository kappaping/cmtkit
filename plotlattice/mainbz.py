## Main function

from math import *
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../lattice')
import lattice as ltc
import brillouinzone as bz
sys.path.append('../bandtheory')
import bandtheory as bdth
import plotband as plbd


ltype='sq'
print(ltc.ltcname(ltype))

prds=[1,1,1]

Nk=6

bzop=True
ks,dks=bz.listbz(ltype,prds,Nk,bzop)

todata=True
dataks=[k+np.array([0.,0.,1.]) for k in ks]
if(len(dks)==2):kcts=[kct for k in ks for kct in [k+dks[0],k+dks[1],k-dks[0],k-dks[1]]]
elif(len(dks)==3):kcts=[kct for k in ks for kct in [k+dks[0],k-dks[2],k+dks[1],k-dks[0],k+dks[2],k-dks[1]]]
dataks+=kcts

#dataks=plbd.fermisurface(H,nf,ltype,uctype,Nk)

filetfig='../../figs/bz.pdf'
tosave=True
tolabel=False
plbd.plotbz(ltype,prds,todata,dataks,tolabel,tosave,filetfig)




