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
ltype='py'
Nbl=[2,2,2]
rs,Nr=ltc.ltcsites(ltype,Nbl)
bc=1
NB,RD,RDV=ltc.ltcpairdist(ltype,rs,Nbl,bc)
# Flavor and state.
Nfl=1
Nrfl=[Nr,Nfl]
Nst=tb.statenum(Nrfl)
# Filling fraction of each state.
nf=1./4.*(1.+(2./8.))
mu=0.

# Tight-binding Hamiltonian.
ts=[0.,-1.,0.]
H=tb.tbham(ts,NB,Nfl)

# Set the unit cell with periodicity prds.
prds=[1,1,1]
rucs,RUCRP=bdth.ftsites(ltype,rs,prds)

hsks=bz.hskpoints(ltype,prds)
print('{')
for hsk in hsks:
    print('{',hsk[1][0],',',hsk[1][1],',',hsk[1][2],'},')
print('}')

# Get the momentum-space Hamiltonian.
Hk=lambda k:bdth.ftham(k,H,Nrfl,RDV,rucs,RUCRP)
Nk=50

filetfig='../../figs/hartreefock/testbd.pdf'
tosave=True
tosetmu=True
plbd.plotbandcontour(Hk,ltype,prds,Nfl,Nk,nf,tosetmu=tosetmu,mu=mu,tosave=tosave,filetfig=filetfig)


