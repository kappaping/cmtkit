## Main function

from math import *
import numpy as np
import time
import joblib

import sys
sys.path.append('../lattice')
import lattice as ltc
sys.path.append('../tightbinding')
import tightbinding as tb
import plotlattice as pltc
import densitymatrix as dm


# Lattice structure
ltype='ka'
print(ltc.ltcname(ltype))
Nbl=[6,6,1]
print('System size = ',Nbl)
Nsl=ltc.slnum(ltype)
rs=ltc.ltcsites(Nbl,Nsl)
Nr=len(rs)
print('Site number = ',Nr)
bc=1
#NB,RD=ltc.ltcpairdist(ltype,rs,Nbl,bc)
#nb1ids=ltc.nthneighbors(1,NB)


filet='../../data/lattice/pyrochlore/888_bc_1'
filetfig='/home/kappaping/research/figs/testfig.pdf'

'''
pltc.plotlattice(rs,Nall,ltype,filetfig,show3d=True)
'''

NB=ltc.ltcpairdist(ltype,rs,Nbl,bc)[0]
#[NB,RD]=joblib.load(filet)
nb1ids=ltc.nthneighbors(1,NB)
pltc.plotlattice(rs,nb1ids,Nbl,ltype,bc,filetfig,show3d=True)


