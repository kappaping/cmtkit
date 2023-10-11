## Main function

from math import *
import numpy as np
import time

import sys
sys.path.append('../lattice')
import lattice as ltc
import tightbinding as tb
import tbsquare as tbsq


ltype='sq'
print(ltc.ltcname(ltype))
Nbl=[3,3,1]
print('[n0,n1,n2] = ',Nbl)
Nsl=ltc.slnum(ltype)
bc=0
rs=ltc.ltcsites(Nbl,Nsl)
Nr=len(rs)
Nfl=2
Nrfl=[Nr,Nfl]
Nst=tb.statenum(Nrfl)
print('State number = ',Nst)

#[print([r,fl],tb.stid(r,fl,Nall)) for r in rs for fl in range(Nfl)]

'''
ts=[0.,-1.,-0.3]

time1=time.time()
NB,RD=ltc.ltcpairdist(ltype,rs,Nbl,bc)
H2=tb.tbham(ts,NB,Nfl)
time2=time.time()
print('time = ',time2-time1)
#print(H2)
'''

'''
utype='co'
NB,RD=ltc.ltcpairdist(ltype,rs,Nbl,bc)
us=[1.,0.5,0.2]
delta=0.5
UINT=tb.dendenint(us,NB,RD,Nfl,utype,delta)
print(UINT)
print(NB)
'''
#'''
t=1.
lda=0.4
m=0.3
print('[t, lambda, m] = ',[t,lda,m])
NB,RD=ltc.ltcpairdist(ltype,rs,Nbl,bc)
nb1ids=ltc.nthneighbors(1,NB)

H=tbsq.spinlessbhz(t,lda,m,rs,nb1ids,Nbl,Nrfl,ltype,bc)

print(H)
#'''
