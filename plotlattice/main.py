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
Nbl=[6,6,1]
rs,Nr=ltc.ltcsites(ltype,Nbl)
bc=1
filet='../../data/lattice/pyrochlore/888_bc_1'
NB,RD,RDV=ltc.ltcpairdist(ltype,rs,Nbl,bc,toread=False,filet=filet)

filetfig='/home/kappaping/research/figs/testfig.pdf'

rids=[n for n in range(len(rs))]
nb1ids=ltc.nthneighbors(1,NB)
otype='wf'
#os=[[2*(0.5-(r[0][0])%2)*(r[1]==1) for r in rs],[],[]]
#os=[[2*(0.5-(r[0][0]-r[0][1])%2)*(r[1]==1) for r in rs],[],[]]
#os=[[((r[0][0]-r[0][1]+1)%3-1)*(r[1]==1) for r in rs],[],[]]
os=[[(1-1.5*np.sign((r[0][0]-r[0][1])%3))*((r[1]==0)-(r[1]==1)) for r in rs],[],[]]
#os=[[2*(0.5-((r[0][0]+r[0][1]+r[0][2])%2))*(r[1]==1) for r in rs],[],[]]
#os=[[2*(0.5-(r[0][0]-r[0][1])%2)*(r[1]==1) for r in rs],[],[]]
res=10
plaz,plel=0.,0.
#plaz,plel=293.,75.
#plaz,plel=255.,69.
pltc.plotlattice(rs,rids,nb1ids,Nbl,ltype,bc,filetfig,otype=otype,os=os,res=res,show3d=False,plaz=plaz,plel=plel)


