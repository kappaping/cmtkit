## Main function

from math import *
import numpy as np
import time
import joblib

import sys
sys.path.append('../common')
import lattice as ltc
import montecarlo as mc




# Lattice setup

mtype=21
fltype=1
print(ltc.ltcname(mtype))
Nbl=[16,16,1]
Nsl=ltc.nslf(mtype)
Nltc=[Nbl,Nsl]
Nr=Nltc[0][0]*Nltc[0][1]*Nltc[0][2]*Nltc[1]
bc=1
rs=ltc.ltcsites(Nltc)

# Monte Carlo setup

NEQ=1000
NMC=2000
Js=[0.,-1.,0.]
Ts=np.linspace(0.01,4.,num=50).tolist()

chis=[]
cTs=[]

for T in Ts:
    time1=time.time()
    fls=mc.latticefl(rs,Nltc,fltype)
    mc.equilibrate(fls,Js,Ts[0],rs,Nltc,Nr,bc,mtype,fltype,NEQ)
    chit,cTt=mc.sampling(fls,Js,T,rs,Nltc,Nr,bc,mtype,fltype,NMC)
    chis.append(chit)
    cTs.append(cTt)
    time2=time.time()
    print('T = ',T,', time = ',time2-time1)

joblib.dump([Ts,chis,cTs],'data_ising_16x16')



