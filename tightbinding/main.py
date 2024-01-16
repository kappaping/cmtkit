## Plot

from math import *
import numpy as np

import sys
sys.path.append('../lattice')
import lattice as ltc
import tightbinding as tb


# Lattice structure.
ltype='ka'
Nbl=[12,12,1]
rs,Nr=ltc.ltcsites(ltype,Nbl)
bc=1
NB,RD,RDV=ltc.ltcpairdist(ltype,rs,Nbl,bc)
# Flavor and state.
Nfl=2
Nrfl=[Nr,Nfl]
# Filling fraction of each state.
nf=1./2.
# Whether to adopt the Bogoliubov-de Gennes form.
tobdg=False

# Tight-binding Hamiltonian.
ts=[0.,-1.,0.]
H=tb.tbham(ts,NB,Nfl)

toprint=True
filetfig='../../figs/hartreefock/testeig.pdf'
tb.plotenergy(H,Nrfl,nf,toprint=toprint,filetfig=filetfig,tobdg=tobdg)


