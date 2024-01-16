## Main function

from math import *
import numpy as np
import matplotlib.pyplot as plt
import joblib

import sys
sys.path.append('../lattice')
import lattice as ltc
sys.path.append('../tightbinding')
import tightbinding as tb
import densitymatrix as dm
import brillouinzone as bz
sys.path.append('../bandtheory')
import bandtheory as bdth
import plotband as plbd


# Lattice structure.
ltype='tr'
Nbl=[4,4,1]
rs,Nr=ltc.ltcsites(ltype,Nbl)
bc=1
NB,RD,RDV=ltc.ltcpairdist(ltype,rs,Nbl,bc)
# Flavor and state.
Nfl=2
Nrfl=[Nr,Nfl]
Nst=tb.statenum(Nrfl)
# Filling fraction of each state.
nf=3./4.-1./8.

# Tight-binding Hamiltonian.
ts=[0.,-1.,0.6]
H=tb.tbham(ts,NB,Nfl)

sys.stdout.flush()

prds=[1,1,1]
rucs,RUCRP=bdth.ftsites(ltype,rs,prds)

Hk=lambda k:bdth.ftham(k,H,Nrfl,RDV,rucs,RUCRP)
Nk=100

bzop=False
ks,dks=bz.listbz(ltype,prds,Nk,bzop)

todata=True
datatype='s'
tosetde=True
de=0.05
dataks=plbd.mapfs(Hk,nf,ltype,prds,Nk,datatype=datatype,tosetde=tosetde,de=de)
nbd=3
#dataks=plbd.mapband(Hk,nbd,ltype,prds,Nk)

filetfig='../../figs/hartreefock/testfs.pdf'
#filetfig='../../figs/hartreefock/testbdbz.pdf'
tosave=True
tolabel=False
plbd.plotbz(ltype,prds,todata,dataks,tolabel=tolabel,tosave=tosave,filetfig=filetfig)




