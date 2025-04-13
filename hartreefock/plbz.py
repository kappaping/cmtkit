## Main function

from math import *
import numpy as np
import matplotlib.pyplot as plt
import joblib

import sys
sys.path.append('../../cmt_code/lattice')
import lattice as ltc
sys.path.append('../../cmt_code/tightbinding')
import tightbinding as tb
import densitymatrix as dm
import brillouinzone as bz
sys.path.append('../../cmt_code/interaction')
import interaction as itn
sys.path.append('../../cmt_code/bandtheory')
import bandtheory as bdth
sys.path.append('../../cmt_code/plotlattice')
import plotband as plbd
import hartreefock as hf


# Lattice structure.
ltype='ka'
Nbl=[2,2,1]
rs,Nr=ltc.ltcsites(ltype,Nbl)
bc=1
filet='../../data/lattice/honeycomb/18181_bc_1'
#filet='../../data/lattice/kagome/12121_bc_1'
#filet='../../data/lattice/pyrochlore/444_bc_1'
NB,RD,RDV=ltc.ltcpairdist(ltype,rs,Nbl,bc,toread=False,filet=filet)
# Flavor and state.
Nfl=1
Nrfl=[Nr,Nfl]
Nst=tb.statenum(Nrfl)
# Filling fraction of each state.
nf=(5./12.)
#nf=(1./3.)*(2.+(3./12.))
# Whether to adopt the Bogoliubov-de Gennes form.
tobdg=False

# Setup of density matrix.
Ptype='read'
filet='../../data/test/test00'
#filet='../../data/hartreefock/bdg/square/nfl_2/fmpsc/nf_12n14_u0_20_u1_n05_8x8_bc_1'
#filet='../../data/hartreefock/bdg/triangular/nfl_1/psc/nf_12_u0_00_u1_n05_12x12_bc_1'
#filet='../../data/hartreefock/bdg/kagome/nfl_1/psc/nf_13p12_u0_00_u1_n05_24x24_bc_1'
#filet='../../data/hartreefock/bdg/kagome/nfl_2/fmpsc/nfb_n12_u0_120_u1_n05_12x12_bc_1'
P=dm.setdenmat(Ptype,Nrfl,nf,fileti=filet,tobdg=tobdg)

# Tight-binding Hamiltonian.
ts=[0.,-1.]
H0=tb.tbham(ts,NB,Nfl)
# Interaction.
us=[0.,4.]
UINT=itn.interaction(NB,Nrfl,us)
# Chemical potential
mu=hf.getchempot(H0,P,UINT,nf,Nst,tobdg=tobdg,toprint=True,toread=True,filet=filet)
H=hf.hfham(H0,P,UINT,tobdg=tobdg,mu=mu)

# Set the unit cell with periodicity prds.
prds=[2,2,1]
rucs,RUCRP=bdth.ftsites(ltype,rs,prds)

# Get the momentum-space Hamiltonian.
Hk=lambda k:bdth.ftham(k,H,Nrfl,RDV,rucs,RUCRP,tobdg=tobdg)

Nk=48

bzop=False
ks,dks=bz.listbz(ltype,prds,Nk,bzop)

todata=True
dBfBfs=[bdth.berrycurv(k,Hk,dks,nf,tobdg=tobdg) for k in ks]
dBfs=[dBfBf[0] for dBfBf in dBfBfs]
Bfs=[dBfBf[1] for dBfBf in dBfBfs]
Ch=(1./(2*pi))*sum(dBfs)
print('Chern number = ',Ch)
data=[Bfs[nk] for nk in range(len(ks))]
dka=dBfBfs[0][2]
bzvol=len(ks)*dka
print('BZ volume = ',bzvol)

#dataks=plbd.fermisurface(H,nf,ltype,uctype,Nk)

filetfig='../../figs/hartreefock/testbz.pdf'
tosave=True
tolabel=False
plbd.plotbz(ltype,prds,ks,todata=todata,data=data,ptype='gd',dks=dks,bzop=bzop,bzvol=bzvol,tolabel=tolabel,tosave=tosave,filetfig=filetfig)




