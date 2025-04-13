## Main function

from math import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors
plt.rcParams['font.size']=18
plt.rcParams.update({'figure.autolayout': True})
import joblib


import sys
sys.path.append('../../cmt_code/lattice')
import lattice as ltc
sys.path.append('../../cmt_code/tightbinding')
import tightbinding as tb
import densitymatrix as dm
sys.path.append('../../cmt_code/interaction')
import interaction as itn
import functionalrg as frg


# Lattice structure.
ltype='tr'
Nbl=[6,6,1]
rs,Nr=ltc.ltcsites(ltype,Nbl)
bc=1
NB,RD,RDV=ltc.ltcpairdist(ltype,rs,Nbl,bc)
# Flavor and state.
Nfl=2
Nrfl=[Nr,Nfl]
Nst=tb.statenum(Nrfl)
# Filling fraction of each state.
nf=3./4.+0./8.

# Version of functional renormalization group: toflsym=True/False if the flavor-symmetric/asymmetric formalism is chosen.
toflsym=True

# If toflsym, construct the model with the flavor dependence removed.
Nflt=Nfl
if(toflsym):Nflt=1
# Tight-binding Hamiltonian.
ts=[0.,-1.,0.]
H0=tb.tbham(ts,NB,Nflt)
prds=[1,1,1]
# Interaction.
us=[4.]
UINT=itn.interaction(NB,[Nr,Nflt],us)

# Maximal neighbor index for the truncation.
nbmax=3
# Cut number of transfer momenta in the Brillouin zone.
Nqc=24
# Cut number of integral momenta in the Brillouin zone.
Nkc=1*Nqc
# Adaptive integral under RG: toadap=True/False if additional points are/aren't added in the integral momentum grids.
toadap=True
# Maximal number of discretization index for integral momentum grids.
Ngmax=0
# Brillouin-zone symmetry: tobzsym=True/False if the symmetry reduction of computation into sub Brillouin zone is/isn't applied for the transfer momenta.
tobzsym=True
# Rotation number for C_Nrot symmetry.
Nrot=6
# Mirror number for mirror symmetry.
Nmir=1


UINTcs,PHIcs,dCHIcs,Tc,xmaxss=frg.functionalrg(UINT,H0,ltype,rs,Nbl,NB,RDV,Nrfl,nf,prds=prds,nbmax=nbmax,Nqc=Nqc,Nkc=Nkc,toflsym=toflsym,toadap=toadap,Ngmax=Ngmax,tobzsym=tobzsym,Nrot=Nrot,Nmir=Nmir)

# File name for writing out.
filet='../../data/test/test10'
# Write out the RG results.
joblib.dump([UINTcs,PHIcs,dCHIcs,Tc,xmaxss],filet)


