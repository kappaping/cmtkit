## Main function

from math import *
import numpy as np
import time

import sys
sys.path.append('../lattice')
import lattice as ltc
import brillouinzone as bz
sys.path.append('../tightbinding')
import tightbinding as tb
import bandtheory as bdth




# Lattice structure.
ltype='ka'
Nbl=[4,4,1]
rs,Nr=ltc.ltcsites(ltype,Nbl)
bc=1
NB,RD,RDV=ltc.ltcpairdist(ltype,rs,Nbl,bc)
# Flavor and state.
Nfl=1
Nrfl=[Nr,Nfl]
Nst=tb.statenum(Nrfl)

# Tight-binding Hamiltonian.
ts=[0.,-1.,0.]
H=tb.tbham(ts,NB,Nfl)

# Set the unit cell with periodicity prds.
prds=[1,1,1]
rucs,RUCRP=bdth.ftsites(ltype,rs,prds)

# Get the momentum-space Hamiltonian.
Hk=lambda k:bdth.ftham(k,H,Nrfl,RDV,rucs,RUCRP)

#k=(bz.hskpoints(ltype,prds)[4][1]+0.0001*(bz.hskpoints(ltype,prds)[2][1]-bz.hskpoints(ltype,prds)[4][1]))
k=(bz.hskpoints(ltype,prds)[4][1]+0.0001*np.array([0.,1.,0.]))
print('k =',k)
eigss=np.linalg.eigh(Hk(k))
for nee in range(eigss[0].shape[0]):print('ee =',eigss[0][nee].round(10),', Vee =',eigss[1][:,nee].round(10))
