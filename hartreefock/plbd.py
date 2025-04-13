## Main function

from math import *
import numpy as np
import joblib

import sys
sys.path.append('../../cmt_code/lattice')
import lattice as ltc
import brillouinzone as bz
sys.path.append('../../cmt_code/tightbinding')
import tightbinding as tb
import densitymatrix as dm
import bogoliubovdegennes as bdg
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
#filet='../../data/lattice/square/16161_bc_1'
filet='../../data/lattice/honeycomb/18181_bc_1'
#filet='../../data/lattice/kagome/12121_bc_1'
#filet='../../data/lattice/pyrochlore/444_bc_1'
NB,RD,RDV=ltc.ltcpairdist(ltype,rs,Nbl,bc,toread=False,filet=filet)
# Flavor and state.
Nfl=1
Nrfl=[Nr,Nfl]
Nst=tb.statenum(Nrfl)
# Filling fraction of each state.
nf=5./12.+(0./8.)
#nf=(1./3.)*(1.+(3./12.))
# Whether to adopt the Bogoliubov-de Gennes form.
tobdg=False

# Setup of density matrix.
Ptype='read'
filet='../../data/test/test00'
#filet='../../data/hartreefock/bdg/square/nfl_2/dsc/nf_12n18_u0_20_u1_n05_16x16_bc_1'
#filet='../../data/hartreefock/bdg/triangular/nfl_1/psc/nf_12_u0_00_u1_n05_12x12_bc_1'
#filet='../../data/hartreefock/bdg/kagome/nfl_1/fsc/nf_13p112_u0_00_u1_n05_24x24_bc_1'
#filet='../../data/hartreefock/bdg/kagome/nfl_2/fmpsc/nfb_n12_u0_120_u1_n05_12x12_bc_1'
#filet='../../data/hartreefock/photokagomecdw/nf512/u1_10_24x24_1'
P=dm.setdenmat(Ptype,Nrfl,nf,fileti=filet,tobdg=tobdg)

# Tight-binding Hamiltonian.
ts=[0.,-1.]
#ts=[(12.*np.array([nf,0.,0.,nf-1.])).tolist(),-1.,0.]
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
'''
kk=bz.hskpoints(ltype,prds)[4]
print('momentum =',kk)
eess=np.linalg.eigh(Hk(kk[1]))
sn=dm.pairspin(P,0,0,Nfl)
sn=sn/np.linalg.norm(sn)
smat=0.5*np.kron(np.identity(ltc.slnum(ltype)),np.tensordot(sn,np.array([tb.paulimat(n) for n in [1,2,3]]),(0,0)))
slmat=np.kron((tb.paulimat(0)+tb.paulimat(3))/2.,np.identity(Nfl))
for n in range(len(eess[0])):print('ee =',eess[0][n],
        ', eevsp =',np.linalg.multi_dot([eess[1][:,n].conj().T,smat,eess[1][:,n]]),
        ', eevsl =',np.linalg.multi_dot([eess[1][:,n].conj().T,slmat,eess[1][:,n]])
        )
'''
filetfig='../../figs/hartreefock/testbd.pdf'
tosave=True
eezm=1
eezmmid=0.68068
zmkts=[0,1]
zmktszm=[1.,-0.98]
plbd.plotbandcontour(Hk,ltype,prds,Nfl,Nk,nf,eezm=eezm,eezmmid=eezmmid,zmkts=zmkts,zmktszm=zmktszm,tosave=tosave,filetfig=filetfig,tobdg=tobdg)


