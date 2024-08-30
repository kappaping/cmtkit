## Main function

from math import *
import numpy as np

import sys
sys.path.append('../lattice')
import lattice as ltc
import brillouinzone as bz
sys.path.append('../tightbinding')
import tightbinding as tb
import densitymatrix as dm
import bogoliubovdegennes as bdg
sys.path.append('../bandtheory')
import bandtheory as bdth
import plotband as plbd


# Lattice structure.
ltype='ka'
Nbl=[2,2,1]
rs,Nr=ltc.ltcsites(ltype,Nbl)
bc=1
filet='../../data/lattice/diamond/888_bc_1'
NB,RD,RDV=ltc.ltcpairdist(ltype,rs,Nbl,bc,toread=False,filet=filet)
# Flavor and state.
Nfl=1
Nrfl=[Nr,Nfl]
Nst=tb.statenum(Nrfl)
# Filling fraction of each state.
nf=5./12.*(1.+(0./8.))
mu=0.

# Tight-binding Hamiltonian.
ts=[0.,-1.]
H=tb.tbham(ts,NB,Nfl)

# Set the unit cell with periodicity prds.
prds=[1,1,1]
rucs,RUCRP=bdth.ftsites(ltype,rs,prds)

# Get the momentum-space Hamiltonian.
Hk=lambda k:bdth.ftham(k,H,Nrfl,RDV,rucs,RUCRP)
Nk=60

filetfig='../../figs/hartreefock/testbd.pdf'
tosave=True
tosetmu=False
plbd.plotbandcontour(Hk,ltype,prds,Nfl,Nk,nf,tosetmu=tosetmu,mu=mu,tosave=tosave,filetfig=filetfig)

hsks=bz.hskpoints(ltype,prds)
for hsk in hsks:
    print('hsk =',hsk)
    ees,eevs=np.linalg.eigh(Hk(hsk[1]))
    print('ees =',ees,', eevs =',eevs.conj().T)




