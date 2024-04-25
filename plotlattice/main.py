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
ltype='ch'
Nbl=[8,8,1]
rs,Nr=ltc.ltcsites(ltype,Nbl)
bc=1
NB,RD,RDV=ltc.ltcpairdist(ltype,rs,Nbl,bc)

#filet='../../data/lattice/pyrochlore/888_bc_1'
filetfig='/home/kappaping/research/figs/testfig.pdf'

rids=[n for n in range(len(rs))]
nb1ids=ltc.nthneighbors(1,NB)
pltc.plotlattice(rs,rids,nb1ids,Nbl,ltype,bc,filetfig,show3d=True)


