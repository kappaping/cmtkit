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
ltype='ka'
Nbl=[2,2,1]
rs,Nr=ltc.ltcsites(ltype,Nbl)
bc=1
filet='../../data/lattice/square/16161_bc_1'
NB,RD,RDV=ltc.ltcpairdist(ltype,rs,Nbl,bc,toread=False,filet=filet)
# Flavor and state.
Nfl=1
Nrfl=[Nr,Nfl]
Nst=tb.statenum(Nrfl)
# Filling fraction of each state.
nf=5./12.
# Whether to adopt the Bogoliubov-de Gennes form.
tobdg=False

# Tight-binding Hamiltonian.
ts=[0.,-1.]
H=tb.tbham(ts,NB,Nfl)

# Set the unit cell with periodicity prds.
prds=[1,1,1]
rucs,RUCRP=bdth.ftsites(ltype,rs,prds)

# Get the momentum-space Hamiltonian.
Hk=lambda k:bdth.ftham(k,H,Nrfl,RDV,rucs,RUCRP,tobdg=tobdg)

Nk=60

bzop=False
ks,dks=bz.listbz(ltype,prds,Nk,bzop)

todata=True
tosetde=True
de=0.01
kps,data=plbd.mapfs(Hk,ks,nf,ltype,prds,Nk,tosetde=tosetde,de=de)

toclmax=False
filetfig='../../figs/hartreefock/testfs.pdf'
tosave=True
tolabel=True
plbd.plotbz(ltype,prds,kps,todata=todata,data=data,ptype='gd',dks=dks,bzop=bzop,toclmax=toclmax,tolabel=tolabel,tosave=tosave,filetfig=filetfig)





