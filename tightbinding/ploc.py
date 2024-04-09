## Main function

from math import *
import numpy as np
import matplotlib.pyplot as plt
import joblib

import sys
sys.path.append('../lattice')
import lattice as ltc
import brillouinzone as bz
sys.path.append('../tightbinding')
import tightbinding as tb
import densitymatrix as dm
sys.path.append('../interaction')
import interaction as itn
sys.path.append('../bandtheory')
import bandtheory as bdth
sys.path.append('../plotlattice')
import plotband as plbd


# Lattice structure.
ltype='ka'
Nbl=[12,12,1]
rs,Nr=ltc.ltcsites(ltype,Nbl)
bc=1
filet='../../data/lattice/kagome/12121_bc_1'
#filet='../../data/lattice/pyrochlore/444_bc_1'
NB,RD,RDV=ltc.ltcpairdist(ltype,rs,Nbl,bc,toread=True,filet=filet)
# Flavor and state.
Nfl=1
Nrfl=[Nr,Nfl]
# Filling fraction of each state.
nf=1./3.*(1.+(6./12.))
#nf=11./12.
# Whether to adopt the Bogoliubov-de Gennes form.
tobdg=False

# Tight-binding Hamiltonian.
ts=[0.,-1.,0.]
H=tb.tbham(ts,NB,Nfl)

# Setup of density matrix.
Nst=tb.statenum(Nrfl)
Noc=round(Nst*nf)
U=np.linalg.eigh(H)[1]
#n0=round(Nst*(0./3.))
#n1=round(Nst*(1./3.))
#P=dm.projdenmat(U,n0,n1,Nst)
P=dm.projdenmat(U,0,Noc,Nst)

# Set the unit cell with periodicity prds.
prds=[1,1,1]

Nkc=12

bzop=False
ks,dks=bz.listbz(ltype,prds,Nkc,bzop)

todata=True
otype='c'
q=2*bz.hskpoints(ltype,prds)[0][1]
print('Transfer momentum q =',q)
nbp=-1
tori='r'
data=dm.formfactor(P,ltype,rs,NB,RDV,Nrfl,ks,q,otype,nbp=nbp,tori=tori,tobdg=tobdg)
print('datamax =',np.max(np.array(data)))

#dataks=plbd.fermisurface(H,nf,ltype,uctype,Nk)

toclmax=True
filetfig='../../figs/hartreefock/testbz.pdf'
tosave=True
tolabel=False
plbd.plotbz(ltype,prds,ks,todata=todata,data=data,ptype='gd',dks=dks,bzop=bzop,toclmax=toclmax,tolabel=tolabel,tosave=tosave,filetfig=filetfig)




