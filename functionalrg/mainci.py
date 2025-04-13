## Main function

from math import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors
plt.rcParams['font.size']=18
plt.rcParams.update({'figure.autolayout': True})
import joblib
import sparse


import sys
sys.path.append('../../cmt_code/lattice')
import lattice as ltc
sys.path.append('../../cmt_code/tightbinding')
import tightbinding as tb
import densitymatrix as dm
sys.path.append('../hartreefock')
import interaction as itn
import functionalrg as frg


# Lattice structure.
ltype='tr'
Nbl=[12,12,1]
rs,Nr=ltc.ltcsites(ltype,Nbl)
bc=1
NB,RD,RDV=ltc.ltcpairdist(ltype,rs,Nbl,bc)
# Flavor and state.
Nfl=2
Nrfl=[Nr,Nfl]
Nst=tb.statenum(Nrfl)
# Filling fraction of each state.
nf=3./4.

# File name for writing out.
filet='../../data/test/test10'
#filet='../../data/functionalrg/uc_u0_40_nbmax_2_8x8_bc_1'

# Tight-binding Hamiltonian.
ts=[0.,-1.,0.]
H0=tb.tbham(ts,NB,Nfl)

[UINTc,Tc,uintsp]=joblib.load(filet)

rid0,rid1=ltc.siteid([[0,0,0],0],rs),ltc.siteid([[0,0,0],0],rs)
print('UINTc[',rs[rid0],rs[rid1],rs[rid1],rs[rid0],'] =\n',np.array([[[[UINTc[tb.stateid(rid0,fl0,Nfl),tb.stateid(rid1,fl1,Nfl),tb.stateid(rid1,fl2,Nfl),tb.stateid(rid0,fl3,Nfl)] for fl3 in range(Nfl)] for fl2 in range(Nfl)] for fl1 in range(Nfl)] for fl0 in range(Nfl)]))
rid0,rid1=ltc.siteid([[0,0,0],0],rs),ltc.siteid([[1,0,0],0],rs)
print('UINTc[',rs[rid0],rs[rid1],rs[rid1],rs[rid0],'] =\n',np.array([[[[UINTc[tb.stateid(rid0,fl0,Nfl),tb.stateid(rid1,fl1,Nfl),tb.stateid(rid1,fl2,Nfl),tb.stateid(rid0,fl3,Nfl)] for fl3 in range(Nfl)] for fl2 in range(Nfl)] for fl1 in range(Nfl)] for fl0 in range(Nfl)]))
rid0,rid1=ltc.siteid([[0,0,0],0],rs),ltc.siteid([[1,0,0],0],rs)
print('UINTc[',rs[rid0],rs[rid1],rs[rid0],rs[rid1],'] =\n',np.array([[[[UINTc[tb.stateid(rid0,fl0,Nfl),tb.stateid(rid1,fl1,Nfl),tb.stateid(rid0,fl2,Nfl),tb.stateid(rid1,fl3,Nfl)] for fl3 in range(Nfl)] for fl2 in range(Nfl)] for fl1 in range(Nfl)] for fl0 in range(Nfl)]))
rid0,rid1=ltc.siteid([[0,0,0],0],rs),ltc.siteid([[1,0,0],0],rs)
print('UINTc[',rs[rid0],rs[rid0],rs[rid1],rs[rid1],'] =\n',np.array([[[[UINTc[tb.stateid(rid0,fl0,Nfl),tb.stateid(rid0,fl1,Nfl),tb.stateid(rid1,fl2,Nfl),tb.stateid(rid1,fl3,Nfl)] for fl3 in range(Nfl)] for fl2 in range(Nfl)] for fl1 in range(Nfl)] for fl0 in range(Nfl)]))
rid0,rid1=ltc.siteid([[0,0,0],0],rs),ltc.siteid([[0,1,0],0],rs)
print('UINTc[',rs[rid0],rs[rid1],rs[rid1],rs[rid0],'] =\n',np.array([[[[UINTc[tb.stateid(rid0,fl0,Nfl),tb.stateid(rid1,fl1,Nfl),tb.stateid(rid1,fl2,Nfl),tb.stateid(rid0,fl3,Nfl)] for fl3 in range(Nfl)] for fl2 in range(Nfl)] for fl1 in range(Nfl)] for fl0 in range(Nfl)]))
rid0,rid1=ltc.siteid([[0,0,0],0],rs),ltc.siteid([[1,1,0],0],rs)
print('UINTc[',rs[rid0],rs[rid1],rs[rid1],rs[rid0],'] =\n',np.array([[[[UINTc[tb.stateid(rid0,fl0,Nfl),tb.stateid(rid1,fl1,Nfl),tb.stateid(rid1,fl2,Nfl),tb.stateid(rid0,fl3,Nfl)] for fl3 in range(Nfl)] for fl2 in range(Nfl)] for fl1 in range(Nfl)] for fl0 in range(Nfl)]))
rid0,rid1=ltc.siteid([[4,4,0],0],rs),ltc.siteid([[4,4,0],0],rs)
print('UINTc[',rs[rid0],rs[rid1],rs[rid1],rs[rid0],'] =\n',np.array([[[[UINTc[tb.stateid(rid0,fl0,Nfl),tb.stateid(rid1,fl1,Nfl),tb.stateid(rid1,fl2,Nfl),tb.stateid(rid0,fl3,Nfl)] for fl3 in range(Nfl)] for fl2 in range(Nfl)] for fl1 in range(Nfl)] for fl0 in range(Nfl)]))


nbmax=3
totrc0=True
prds=[1,1,1]
Nk=48
tocr=False
tocfl=1
UINTc0=frg.criticalinteraction(H0,NB,UINTc,Tc,Nrfl,rs,nf,nbmax=nbmax,tocr=tocr,tocfl=tocfl,totrc0=totrc0,ltype=ltype,Nbl=Nbl,prds=prds,RDV=RDV,Nk=Nk)
filet='../../data/test/test11'
#filet='../../data/functionalrg/uc0_u0_40_nbmax_2_8x8_bc_1'
joblib.dump(UINTc0,filet)
#'''
#UINTc0=joblib.load(filet)
rid0,rid1=ltc.siteid([[0,0,0],0],rs),ltc.siteid([[0,0,0],0],rs)
print('UINTc0[',rs[rid0],rs[rid1],rs[rid1],rs[rid0],'] =\n',np.array([[[[UINTc0[tb.stateid(rid0,fl0,Nfl),tb.stateid(rid1,fl1,Nfl),tb.stateid(rid1,fl2,Nfl),tb.stateid(rid0,fl3,Nfl)] for fl3 in range(Nfl)] for fl2 in range(Nfl)] for fl1 in range(Nfl)] for fl0 in range(Nfl)]))
rid0,rid1=ltc.siteid([[0,0,0],0],rs),ltc.siteid([[1,0,0],0],rs)
print('UINTc0[',rs[rid0],rs[rid1],rs[rid1],rs[rid0],'] =\n',np.array([[[[UINTc0[tb.stateid(rid0,fl0,Nfl),tb.stateid(rid1,fl1,Nfl),tb.stateid(rid1,fl2,Nfl),tb.stateid(rid0,fl3,Nfl)] for fl3 in range(Nfl)] for fl2 in range(Nfl)] for fl1 in range(Nfl)] for fl0 in range(Nfl)]))
rid0,rid1=ltc.siteid([[0,0,0],0],rs),ltc.siteid([[1,0,0],0],rs)
print('UINTc0[',rs[rid0],rs[rid1],rs[rid0],rs[rid1],'] =\n',np.array([[[[UINTc0[tb.stateid(rid0,fl0,Nfl),tb.stateid(rid1,fl1,Nfl),tb.stateid(rid0,fl2,Nfl),tb.stateid(rid1,fl3,Nfl)] for fl3 in range(Nfl)] for fl2 in range(Nfl)] for fl1 in range(Nfl)] for fl0 in range(Nfl)]))
rid0,rid1=ltc.siteid([[0,0,0],0],rs),ltc.siteid([[1,0,0],0],rs)
print('UINTc0[',rs[rid0],rs[rid0],rs[rid1],rs[rid1],'] =\n',np.array([[[[UINTc0[tb.stateid(rid0,fl0,Nfl),tb.stateid(rid0,fl1,Nfl),tb.stateid(rid1,fl2,Nfl),tb.stateid(rid1,fl3,Nfl)] for fl3 in range(Nfl)] for fl2 in range(Nfl)] for fl1 in range(Nfl)] for fl0 in range(Nfl)]))
rid0,rid1=ltc.siteid([[0,0,0],0],rs),ltc.siteid([[0,1,0],0],rs)
print('UINTc0[',rs[rid0],rs[rid1],rs[rid1],rs[rid0],'] =\n',np.array([[[[UINTc0[tb.stateid(rid0,fl0,Nfl),tb.stateid(rid1,fl1,Nfl),tb.stateid(rid1,fl2,Nfl),tb.stateid(rid0,fl3,Nfl)] for fl3 in range(Nfl)] for fl2 in range(Nfl)] for fl1 in range(Nfl)] for fl0 in range(Nfl)]))
rid0,rid1=ltc.siteid([[0,0,0],0],rs),ltc.siteid([[1,1,0],0],rs)
print('UINTc0[',rs[rid0],rs[rid1],rs[rid1],rs[rid0],'] =\n',np.array([[[[UINTc0[tb.stateid(rid0,fl0,Nfl),tb.stateid(rid1,fl1,Nfl),tb.stateid(rid1,fl2,Nfl),tb.stateid(rid0,fl3,Nfl)] for fl3 in range(Nfl)] for fl2 in range(Nfl)] for fl1 in range(Nfl)] for fl0 in range(Nfl)]))
rid0,rid1=ltc.siteid([[4,4,0],0],rs),ltc.siteid([[4,4,0],0],rs)
print('UINTc0[',rs[rid0],rs[rid1],rs[rid1],rs[rid0],'] =\n',np.array([[[[UINTc0[tb.stateid(rid0,fl0,Nfl),tb.stateid(rid1,fl1,Nfl),tb.stateid(rid1,fl2,Nfl),tb.stateid(rid0,fl3,Nfl)] for fl3 in range(Nfl)] for fl2 in range(Nfl)] for fl1 in range(Nfl)] for fl0 in range(Nfl)]))
#'''



