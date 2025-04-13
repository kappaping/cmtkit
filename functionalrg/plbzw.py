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
Nfl=4
Nrfl=[Nr,Nfl]
Nst=tb.statenum(Nrfl)
# Filling fraction of each state.
nf=3./4.+0./8.

# File name for writing out.
filet='../../data/test/test10'
#filet='../../data/functionalrg/triangular/nf_34_u0_40_nbmax_3_nqc_24'

nbmax=3
prds=[1,1,1]
Nkc=24
Nqc=24

bzop=True
qs,dqs=bz.listbz(ltype,prds,Nqc,bzop)
Nq=len(qs)
print('Number of points =',Nq)
Nrot=6
Nmir=1
qis,qc=frg.listsbz(ltype,prds,Nrot,Nmir,qs,tobzsym=True)
Nqi=len(qis)
qqiids=frg.sbzids(qs,qis,qc,Nrot,Nmir)

UINTcs,PHIcs,dCHIcs,Tc,phimaxss=joblib.load(filet)

bzop=False
ks,dks=bz.listbz(ltype,prds,Nkc,bzop)
Nk=len(ks)
print('Number of points =',Nk)
toflsym=True
chtype=2
toct=True
nwps=[0]
cwps=[1.]
tosetq=0
idps=[0,0]
reim=0
#data=frg.leadinginstability(UINTcs,dCHIcs,qis,ks,ltype,rs,Nbl,NB,nbmax,prds,Nfl,toflsym=toflsym,chtype=chtype,toct=toct,nwps=nwps,cwps=cwps,tosetq=tosetq,idps=idps,reim=reim)*100
data=frg.leadinginstability(PHIcs,dCHIcs,qis,ks,ltype,rs,Nbl,NB,nbmax,prds,Nfl,toflsym=toflsym,chtype=chtype,toct=toct,nwps=nwps,cwps=cwps,tosetq=tosetq,idps=idps,reim=reim)*100

filetfig='../../figs/bz.pdf'
todata=True
tosave=True
tolabel=False
plbd.plotbz(ltype,prds,ks,todata=todata,data=data,ptype='gd',dks=dks,bzop=bzop,tolabel=tolabel,tosave=tosave,filetfig=filetfig)




