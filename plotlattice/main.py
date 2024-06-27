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
ltype='fcc0'
Nbl=[2,2,2]
rs,Nr=ltc.ltcsites(ltype,Nbl)
bc=1
filet='../../data/lattice/diamond/444_bc_1'
NB,RD,RDV=ltc.ltcpairdist(ltype,rs,Nbl,bc,toread=False,filet=filet)

filetfig='/home/kappaping/research/figs/testfig.pdf'

rids=[n for n in range(len(rs))]
nb1ids=ltc.nthneighbors(1,NB)

res=10
plaz,plel=0.,0.
dist=None
pltc.plotlattice(rs,rids,nb1ids,Nbl,ltype,bc,filetfig,res=res,show3d=True,plaz=plaz,plel=plel,dist=dist)


