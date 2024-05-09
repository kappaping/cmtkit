## Plot

from math import *
import numpy as np

import sys
sys.path.append('../lattice')
import lattice as ltc
import tightbinding as tb


# Lattice structure.
ltype='ka'
Nbl=[24,24,1]
rs,Nr=ltc.ltcsites(ltype,Nbl)
bc=1
filet='../../data/lattice/kagome/24241_bc_1'
#filet='../../data/lattice/pyrochlore/444_bc_1'
NB,RD,RDV=ltc.ltcpairdist(ltype,rs,Nbl,bc,toread=True,filet=filet)
# Flavor and state.
Nfl=1
Nrfl=[Nr,Nfl]
# Filling fraction of each state.
nf=1./3.*(1.+(0./12.))
# Whether to adopt the Bogoliubov-de Gennes form.
tobdg=False

# Tight-binding Hamiltonian.
ts=[0.,-1.,0.]
H=tb.tbham(ts,NB,Nfl)

toprint=True
filetfig='../../figs/hartreefock/testeig.pdf'
tb.plotenergy(H,Nrfl,nf,toprint=toprint,filetfig=filetfig,tobdg=tobdg)


