## Main function

from math import *
import numpy as np
import time

import sys
sys.path.append('../lattice')
import lattice as ltc
import tightbinding as tb
import lattice2 as ltc2
import tightbinding2 as tb2
import tbsquare as tbsq


ltype='sq'
print(ltc.ltcname(ltype))
Nbl=[3,3,1]
print('[n0,n1,n2] = ',Nbl)
Nsl=ltc.slnum(ltype)
bc=0
rs=ltc2.ltcsites(Nbl,Nsl)
Nr=len(rs)
Nfl=2
Nall=[[Nbl,Nsl],Nfl]
Nrfl=[Nr,Nfl]
Nst=tb2.statenum(Nrfl)
print('State number = ',Nst)

#[print([r,fl],tb.stid(r,fl,Nall)) for r in rs for fl in range(Nfl)]

'''
ts=[0.,-1.,-0.3]

time1=time.time()
H=tb.tbham(ts,rs,Nall,bc,ltype)
time2=time.time()
print('time = ',time2-time1)
#print(H)


time1=time.time()
NB,RD=ltc2.ltcpairdist(ltype,rs,Nbl,bc)
H2=tb2.tbham(ts,NB,Nfl)
time2=time.time()
print('time = ',time2-time1)
#print(H2)

print('max diff = ',np.max(np.abs(H-H2)))
'''

'''
utype='co'
NB,RD=ltc2.ltcpairdist(ltype,rs,Nbl,bc)
us=[1.,0.5,0.2]
delta=0.5
UINT=tb2.dendenint(us,NB,RD,Nfl,utype,delta)
print(UINT)
print(NB)
'''
#'''
t=1.
lda=0.4
m=0.3
print('[t, lambda, m] = ',[t,lda,m])
NB,RD=ltc2.ltcpairdist(ltype,rs,Nbl,bc)
nb1ids=ltc2.nthneighbors(1,NB)

H=tbsq.spinlessbhz(t,lda,m,rs,nb1ids,Nbl,Nrfl,ltype,bc)

print(H)
#'''
