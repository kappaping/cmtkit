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


ltype='tr'
print(ltc.ltcname(ltype))

prds=[1,1,1]

Nkc=30

bzop=True
ks,dks=bz.listbz(ltype,prds,Nkc,bzop)
print('Number of points =',len(ks))

todata=True
data=[1. for k in ks]
#if(len(dks)==2):kcts=[kct for k in ks for kct in [k+dks[0],k+dks[1],k-dks[0],k-dks[1]]]
#elif(len(dks)==3):kcts=[kct for k in ks for kct in [k+dks[0],k-dks[2],k+dks[1],k-dks[0],k+dks[2],k-dks[1]]]
#dataks+=kcts

#dataks=plbd.fermisurface(H,nf,ltype,uctype,Nk)

filetfig='../../figs/bz.pdf'
tosave=True
tolabel=True
bzop=True
plbd.plotbz(ltype,prds,ks,todata=todata,data=data,ptype='gd',dks=dks,bzop=bzop,tolabel=tolabel,tosave=tosave,filetfig=filetfig)




