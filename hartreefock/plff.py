## Main function

from math import *
import numpy as np
import matplotlib.pyplot as plt
import joblib

import sys
sys.path.append('../../cmt_code/lattice')
import lattice as ltc
import brillouinzone as bz
sys.path.append('../../cmt_code/tightbinding')
import tightbinding as tb
import densitymatrix as dm
sys.path.append('../../cmt_code/interaction')
import interaction as itn
sys.path.append('../../cmt_code/bandtheory')
import bandtheory as bdth
sys.path.append('../../cmt_code/plotlattice')
import plotband as plbd
import hartreefock as hf


# Lattice structure.
ltype='ka'
Nbl=[12,12,1]
rs,Nr=ltc.ltcsites(ltype,Nbl)
bc=1
#filet='../../data/lattice/square/16161_bc_1'
#filet='../../data/lattice/triangular/12121_bc_1'
filet='../../data/lattice/kagome/12121_bc_1'
#filet='../../data/lattice/pyrochlore/444_bc_1'
NB,RD,RDV=ltc.ltcpairdist(ltype,rs,Nbl,bc,toread=True,filet=filet)
# Flavor and state.
Nfl=2
Nrfl=[Nr,Nfl]
Nst=tb.statenum(Nrfl)
# Filling fraction of each state.
nf=3./12.-0./16.
# Whether to adopt the Bogoliubov-de Gennes form.
tobdg=True

# Setup of density matrix.
Ptype='copy'
filet='../../data/test/test05'
#filet='../../data/hartreefock/bdg/square/nfl_2/dsc/nf_12n18_u0_20_u1_n05_16x16_bc_1'
#filet='../../data/hartreefock/bdg/triangular/nfl_1/psc/nf_12_u0_00_u1_n05_12x12_bc_1'
#filet='../../data/hartreefock/bdg/kagome/nfl_1/psc/nf_13p12_u0_00_u1_n05_24x24_bc_1'
filet='../../data/hartreefock/bdg/kagome/nfl_2/fmpsc/nfb_n12_u0_120_u1_n05_12x12_bc_1'
P=dm.setdenmat(Ptype,Nrfl,nf,tobdg=tobdg,fileti=filet,ltype=ltype,rs=rs,Nbl=Nbl,NB=NB,nbcpmax=1,Nbli=Nbl)

# Tight-binding Hamiltonian.
ts=[0.,-1.,0.]
#ts=[(12.*np.array([nf,0.,0.,nf-1.])).tolist(),-1.,0.]
H0=tb.tbham(ts,NB,Nfl)
H=H0

# Set the unit cell with periodicity prds.
prds=[1,1,1]
rucs,RUCRP=bdth.ftsites(ltype,rs,prds)

# Get the momentum-space Hamiltonian.
Hk=lambda k:bdth.ftham(k,H,Nrfl,RDV,rucs,RUCRP)
Nkc=Nbl[0]

bzop=False
ks,dks=bz.listbz(ltype,prds,Nkc,bzop)

todata=True
otype='fe'
q=bz.hskpoints(ltype,prds)[0][1]
print('Transfer momentum q =',q)
nbds=[1,1]
nbp=-1
tori='i'
data=dm.formfactor(P,Hk,ltype,rs,NB,RDV,Nrfl,rucs,RUCRP,ks,bzop,q,otype,nbds=nbds,nbp=nbp,tori=tori,tobdg=tobdg)
print('datamax =',np.max(np.array(data)))

#dataks=plbd.fermisurface(H,nf,ltype,uctype,Nk)

toclmax=True
filetfig='../../figs/hartreefock/testbz.pdf'
tosave=True
tolabel=False
plbd.plotbz(ltype,prds,ks,todata=todata,data=data,ptype='gd',dks=dks,bzop=bzop,toclmax=toclmax,tolabel=tolabel,tosave=tosave,filetfig=filetfig)




