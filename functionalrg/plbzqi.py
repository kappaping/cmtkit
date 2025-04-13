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
sys.path.append('../../cmt_code/bandtheory')
import bandtheory as bdth
sys.path.append('../../cmt_code/plotlattice')
import plotband as plbd
sys.path.append('../hartreefock')
import hartreefock as hf
import interaction as itn
import functionalrg as frg


# Lattice structure.
ltype='tr'
Nbl=[6,6,1]
rs,Nr=ltc.ltcsites(ltype,Nbl)
bc=1
NB,RD,RDV=ltc.ltcpairdist(ltype,rs,Nbl,bc)
# Flavor and state.
Nfl=1
Nrfl=[Nr,Nfl]
Nst=tb.statenum(Nrfl)
# Filling fraction of each state.
nf=3./4.+0./8.

# File name for writing out.
filet='../../data/test/test10'

# Tight-binding Hamiltonian.
ts=[0.,-1.,0.]
H0=tb.tbham(ts,NB,Nfl)
# Interaction.
us=[4.]
jhe1=0./2.
jhes=[0.,jhe1,jhe1*(ts[2]/ts[1])**2]
UINT=itn.interaction(NB,Nrfl,us,jhes=jhes)

nbmax=3
prds=[1,1,1]
Nkc=24
Nqc=24

bzop=True
ks,dks=bz.listbz(ltype,prds,Nkc,bzop)
Nk=len(ks)
print('Number of points =',Nk)
qs,dqs=bz.listbz(ltype,prds,Nqc,bzop)
Nq=len(qs)
print('Number of points =',Nq)
Nrot=6
Nmir=1
qis,qc=frg.listsbz(ltype,prds,Nrot,Nmir,qs,tosym=True)
Nqi=len(qis)
qqiids=frg.sbzids(qs,qis,qc,Nrot,Nmir)

chipm=1
qa=np.array([np.linalg.norm(q-np.array([-2.*pi/sqrt(3),0,0])) for q in qs])
#nq=np.argwhere(qa<1e-14)[0,0]


todata=True

data=[qqiid[0] for qqiid in qqiids]
#data=[1 for qi in qis]
#data=[int(qqiid[1][0]==0 and qqiid[1][1]==0) for qqiid in qqiids]

filetfig='../../figs/bz.pdf'
tosave=True
tolabel=False
bzop=True
#plbd.plotbz(ltype,prds,ks,todata=todata,data=data,ptype='gd',dks=dks,bzop=bzop,tolabel=tolabel,tosave=tosave,filetfig=filetfig)
#plbd.plotbz(ltype,prds,qis,todata=todata,data=data,ptype='gd',dks=dks,bzop=bzop,tolabel=tolabel,tosave=tosave,filetfig=filetfig)
plbd.plotbz(ltype,prds,qs,todata=todata,data=data,ptype='gd',dks=dqs,bzop=bzop,tolabel=tolabel,tosave=tosave,filetfig=filetfig)




