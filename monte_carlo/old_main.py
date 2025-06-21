## Main function

from math import *
import numpy as np
import time
import joblib

import sys
sys.path.append('../lattice')
import lattice as ltc
import montecarlo as mc




# Lattice setup

ltype='sq'
fltype=3
print(ltc.ltcname(ltype))
Nbl=[16,16,1]
Nsl=ltc.slnum(ltype)
Nltc=[Nbl,Nsl]
Nr=Nltc[0][0]*Nltc[0][1]*Nltc[0][2]*Nltc[1]
bc=1
rs=ltc.ltcsites(Nltc)

# Monte Carlo setup

NEQ=200
NMC=1000
fm=-1
Js=[0.,1.,0.]
Ts=np.linspace(0.01,4.,num=20).tolist()

chis=[]
cTs=[]

for T in Ts:
    time1=time.time()
    tofm=0
    fls=mc.latticefl(rs,Nltc,fltype,tofm)
    mc.equilibrate(fls,Js,Ts[0],rs,Nltc,Nr,bc,ltype,fltype,NEQ)
    chit,cTt=mc.sampling(fls,Js,T,rs,Nltc,Nr,bc,ltype,fltype,fm,NMC)
    chis.append(chit)
    cTs.append(cTt)
    time2=time.time()
    print('T = ',T,', time = ',time2-time1)

joblib.dump([Ts,chis,cTs],'data_heisenberg_16x16')



