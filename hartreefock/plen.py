## Plot

from math import *
import numpy as np

import sys
sys.path.append('../../cmt_code/lattice')
import lattice as ltc
sys.path.append('../../cmt_code/tightbinding')
import tightbinding as tb
import densitymatrix as dm
sys.path.append('../../cmt_code/interaction')
import interaction as itn
import hartreefock as hf


# Lattice structure.
ltype='tr'
Nbl=[12,12,1]
rs,Nr=ltc.ltcsites(ltype,Nbl)
bc=1
NB,RD,RDV=ltc.ltcpairdist(ltype,rs,Nbl,bc)
# Flavor and state.
Nfl=1
Nrfl=[Nr,Nfl]
# Filling fraction of each state.
nf=7./12.
# Whether to adopt the Bogoliubov-de Gennes form.
tobdg=False

# Setup of density matrix.
Ptype='read'
filet='../../data/test/test00'
#filet='../../data/hartreefock/bdg/sq_nf_12_u0_40_8x8_bc_1'
P=dm.setdenmat(Ptype,Nrfl,nf,fileti=filet,tobdg=tobdg)

# Tight-binding Hamiltonian.
ts=[0.,-1.,0.]
H0=tb.tbham(ts,NB,Nfl)
# Interaction.
us=[0.]
Jhe=0.
UINT=itn.interaction(NB,Nrfl,us,Jhe=Jhe)

# Get the Hartree-Fock Hamiltonian.
Nst=tb.statenum(Nrfl)
Noc=round(Nst*nf)
mu=hf.getchempot(H0,P,UINT,nf,Nst,tobdg=tobdg,toprint=True,toread=True,filet=filet)
Hhf=hf.hfham(H0,P,UINT,tobdg=tobdg,mu=mu)

toprint=True
filetfig='../../figs/hartreefock/testeig.pdf'
tb.plotenergy(Hhf,Nrfl,nf,toprint=toprint,filetfig=filetfig,tobdg=tobdg)


