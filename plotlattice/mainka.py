## Main function

from math import *
import cmath
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../lattice')
import lattice as ltc
import brillouinzone as bz
sys.path.append('../tightbinding')
import densitymatrix as dm
sys.path.append('../bandtheory')
import bandtheory as bdth
import plotband as plbd


ltype='ka'
print(ltc.ltcname(ltype))

prds=[1,1,1]

Nkc=24

bzop=False
ks,dks=bz.listbz(ltype,prds,Nkc,bzop)
print('Number of points =',len(ks))

def vee(k):
    k0,k1=k[0],k[1]
    v000=-(1+cos(2*k1)-cmath.sqrt(3+4*cos(sqrt(3)*k0)*cos(k1)+2*cos(2*k1))+cos(sqrt(3)*k0+k1))
    v001=(cos((sqrt(3)*k0+k1)/2)*(-1+cmath.sqrt(3+4*cos(sqrt(3)*k0)*cos(k1)+2*cos(2*k1)))-2*cos((sqrt(3)*k0-k1)/2)*cos(k1))
    v010=(cos(sqrt(3)*k0)-cos(k1)*(-2+cmath.sqrt(3+4*cos(sqrt(3)*k0)*cos(k1)+2*cos(2*k1))))
    v011=(cos((sqrt(3)*k0-3*k1)/2)-cos((sqrt(3)*k0+k1)/2)*(-2+cmath.sqrt(3+4*cos(sqrt(3)*k0)*cos(k1)+2*cos(2*k1))))
    v0=np.array([v000*v011,v010*v001,v001*v011])
    nv0=np.linalg.norm(v0)
    if(nv0<1e-14):v0n=np.array([0.,0.,0.])
    else:v0n=v0/nv0
    return v0n

veeks=[vee(k) for k in ks]

q=-bz.hskpoints(ltype,prds)[4][1]
print('Transfer momentum q =',q)
kqids=dm.momentumpairs(ks,[q],ltype,[1,1,1],-1)

todata=True
#data=[vee(k)[0].real for k in ks]
data=[(veeks[kqids[0,kid]][0]*veeks[kid][1]).real for kid in range(len(ks))]
print('datamax =',np.max(np.abs(np.array(data))))

filetfig='../../figs/bz.pdf'
tosave=True
tolabel=False
toclmax=True
plbd.plotbz(ltype,prds,ks,todata=todata,data=data,ptype='gd',dks=dks,bzop=bzop,toclmax=toclmax,tolabel=tolabel,tosave=tosave,filetfig=filetfig)




