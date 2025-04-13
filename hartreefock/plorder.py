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
import bogoliubovdegennes as bdg
sys.path.append('../../cmt_code/plotlattice')
import plotlattice as pltc


# Lattice structure.
ltype='sq'
Nbl=[16,16,1]
rs,Nr=ltc.ltcsites(ltype,Nbl)
bc=1
filet='../../data/lattice/square/16161_bc_1'
#filet='../../data/lattice/checkerboard/32321_bc_1'
#filet='../../data/lattice/lieb/24241_bc_1'
#filet='../../data/lattice/triangular/12121_bc_1'
#filet='../../data/lattice/honeycomb/30301_bc_1'
#filet='../../data/lattice/kagome/12121_bc_1'
#filet='../../data/lattice/bcc0/888_bc_1'
#filet='../../data/lattice/fcc0/888_bc_1'
#filet='../../data/lattice/diamond/888_bc_1'
#filet='../../data/lattice/pyrochlore/888_bc_1'
NB,RD,RDV=ltc.ltcpairdist(ltype,rs,Nbl,bc,toread=True,filet=filet)
# Flavor and state.
Nfl=2
Nrfl=[Nr,Nfl]
# Filling fraction of each state.
nf=1./8.
# Whether to adopt the Bogoliubov-de Gennes form.
tobdg=False

# Setup of density matrix.
Ptype='read'
filet='../../data/test/test00'
#filet='../../data/slp/hf/fcc/u0_00_u1_05_8x8x8'
#filet='../../data/hartreefock/bdg/square/nfl_2/fmpsc/nf_12n14_u0_20_u1_n05_16x16_bc_1'
#filet='../../data/hartreefock/bdg/triangular/nfl_1/psc/nf_12_u0_00_u1_n05_12x12_bc_1'
#filet='../../data/hartreefock/bdg/kagome/nfl_1/psc/nf_12_u0_00_u1_n01_12x12_bc_1'
#filet='../../data/hartreefock/bdg/kagome/nfl_2/fmpsc/nfb_n12_u0_120_u1_n05_12x12_bc_1'
#filet='../../data/hartreefock/photokagomecdw/nf512/u1_10_24x24_1'
toflrot=False
P=dm.setdenmat(Ptype,Nrfl,nf,fileti=filet,ltype=ltype,rs=rs,NB=NB,RDV=RDV,toflrot=toflrot,tobdg=tobdg)

# Plot the orders.
rpls=[]
#rpls=[[[n0,n1,n2],sl] for n0 in [0] for n1 in [0] for n2 in [0] for sl in range(ltc.slnum(ltype))]
#rpls=[[[n0,n1,0],sl] for n0 in range(12) for n1 in range(12) for sl in range(ltc.slnum(ltype))]
#rpls=[[[n0,n1,n2],sl] for n0 in range(4) for n1 in range(4) for n2 in range(4) for sl in range(ltc.slnum(ltype))]
#rpls=[[R,sl] for R in [[0,0,0],[1,0,0],[0,1,0]] for sl in range(ltc.slnum(ltype))]
#rpls=[[[n0,n1,0],0] for n0 in [10,11] for n1 in [0,1]]
Nnb=2
scl=1
res=10
#res=50
show3d=False
plaz,plel,dist=0.,0.,None
#plaz,plel,dist=0.,0.,6.5 # li
#plaz,plel,dist=295.,90.,5. # bcc
#plaz,plel,dist=105.,80.,5.5 # fcc
#plaz,plel,dist=255.,69.,None # dia
#plaz,plel,dist=293.,75.,None # py
filetfigc='/home/kappaping/research/figs/hartreefock/testfigc.pdf'
filetfigs='/home/kappaping/research/figs/hartreefock/testfigs.pdf'
filetfigo='/home/kappaping/research/figs/hartreefock/testfigo.pdf'
filetfigfe='/home/kappaping/research/figs/hartreefock/testfigfe.pdf'
filetfigfo='/home/kappaping/research/figs/hartreefock/testfigfo.pdf'
filetfig=[[filetfigc,filetfigs],[filetfigfe,filetfigfo]]

pltc.plotorder(P,ltype,rs,Nrfl,Nbl,bc,NB,rpls=rpls,Nnb=Nnb,scl=scl,res=res,show3d=show3d,plaz=plaz,plel=plel,dist=dist,filetfig=filetfig,tobdg=tobdg)

