## Main function

from math import *
import numpy as np
import joblib


import sys
sys.path.append('../../cmt_code/lattice')
import lattice as ltc
sys.path.append('../../cmt_code/tightbinding')
import tightbinding as tb
import densitymatrix as dm
sys.path.append('../../cmt_code/plotlattice')
import plotlattice as pltc
import bandtheory as bdth


# Lattice structure
ltype='ka'
print(ltc.ltcname(ltype))
Nbl=[6,6,1]
#Nbl=[12,12,1]
#Nbl=[24,24,1]
#Nbl=[2,2,2]
#Nbl=[4,4,4]
#Nbl=[6,6,6]
print('System size = ',Nbl)
Nsl=ltc.slnum(ltype)
rs=ltc.ltcsites(Nbl,Nsl)
Nr=len(rs)
print('Site number = ',Nr)
bc=1
print('Boundary condition = ',bc)
NB,RD=ltc.ltcpairdist(ltype,rs,Nbl,bc)
#[NB,RD]=joblib.load('../../data/lattice/pyrochlore/888_bc_1')
nb1ids=ltc.nthneighbors(1,NB)

# Flavor and state
Nfl=1
Nrfl=[Nr,Nfl]
Nst=tb.statenum(Nrfl)

P=np.zeros((Nst,Nst))

chos=dm.chargeorder(P,nb1ids,Nrfl)[0]

prds=[223,223,1]
Nuc,rucs=bdth.ucsites(ltype,prds)
chos[0]=[bdth.ucsiteid(r,prds,Nuc,rucs) for r in rs]
cavg=sum(chos[0])/len(chos[0])
chos[0]=[cho-cavg for cho in chos[0]]
cmax=max(chos[0])
chos[0]=[cho/cmax for cho in chos[0]]

to3d=True
show3d=True
dpit=300
#res=50
res=10
plaz,plel=0.,0.
#plaz,plel=295.,75.
filetfigc='/home/kappaping/research/figs/hartreefock/testfigc.pdf'
pltc.plotlattice(rs,nb1ids,Nbl,ltype,bc,filetfigc,'c',chos,res=res,setdpi=dpit,to3d=to3d,show3d=show3d,plaz=plaz,plel=plel)




