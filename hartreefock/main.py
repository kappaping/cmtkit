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
sys.path.append('../../cmt_code/interaction')
import interaction as itn
import hartreefock as hf


# Lattice structure.
ltype='sq'
Nbl=[16,16,1]
rs,Nr=ltc.ltcsites(ltype,Nbl)
bc=1
filet='../../data/lattice/square/16161_bc_1'
#filet='../../data/lattice/checkerboard/32321_bc_1'
#filet='../../data/lattice/lieb/24241_bc_1'
#filet='../../data/lattice/triangular/12121_bc_1'
#filet='../../data/lattice/honeycomb/18181_bc_1'
#filet='../../data/lattice/kagome/18181_bc_1'
#filet='../../data/lattice/bcc0/888_bc_1'
#filet='../../data/lattice/fcc0/888_bc_1'
#filet='../../data/lattice/diamond/888_bc_1'
#filet='../../data/lattice/pyrochlore/888_bc_1'
NB,RD,RDV=ltc.ltcpairdist(ltype,rs,Nbl,bc,toread=True,filet=filet)
# Flavor and state.
Nfl=2
Nrfl=[Nr,Nfl]
Nst=tb.statenum(Nrfl)
# Filling fraction of each state.
nf=(1./2.)+(0./1.)
#nf=(1./3.)*(1.+(3./12.)) # ka
#nf=(1./4.)*(1.+(4./8.)) # fcc
#nf=(1./4.)*(1.+(5./8.)) # py
# Whether to adopt the Bogoliubov-de Gennes form.
tobdg=False

sys.stdout.flush()

# File name for writing out the density matrix.
filet='../../data/test/test00'

# Setup of initial density matrix.
Ptype='rand'
fileti='../../data/test/test01'
#fileti='../../data/slp/hf/fcc/u0_00_u1_05_8x8x8'
#fileti='../../data/hartreefock/bdg/triangular/nfl_1_nf_16_u0_00_u1_n01_12x12_bc_1'
#fileti='../../data/hartreefock/bdg/square/nfl_2/dsc/nf_12n18_u0_20_u1_n05_8x8_bc_1'
#fileti='../../data/hartreefock/kagome/nfbn23/120sdw/t2_00_u0_60_u1_00_u2_00_12x12'
#fileti='../../data/hartreefock/kagome/nfbn1/fm/t2_00_u0_120_u1_00_u2_00_12x12'
toptb=True
toflrot=False
Pi=dm.setdenmat(Ptype,Nrfl,nf,tobdg=tobdg,fileti=fileti,toptb=toptb,toflrot=toflrot,ltype=ltype,rs=rs,Nbl=Nbl,NB=NB,RDV=RDV,Nbli=[12,12,1])

sys.stdout.flush()

# Tight-binding Hamiltonian.
ts=[0.,-1.]
#ts=[(12.*np.array([nf,0.,0.,nf-1.])).tolist(),-1.,0.]
H0=tb.tbham(ts,NB,Nfl)
# Interaction.
us=[4.]
UINT=itn.interaction(NB,Nrfl,us)
#filetitn='../../data/test/test11'
#UINT=joblib.load(filetitn)
# Chemical potential
mu=hf.getchempot(H0,Pi,UINT,nf,Nst,tobdg=tobdg,dnf0=1./Nst**2,toprint=True,toread=False,filet=fileti)

sys.stdout.flush()

# Algorithm: Set the parameters and run the computation.
tofile=True
optm=1
Nnb=2
printdm=20
writedm=40
Nhf=1000000
Nhfm=10

Pf=hf.hartreefock(Pi,H0,UINT,NB,Nrfl,nf,tofile=tofile,filet=filet,optm=optm,Nnb=Nnb,printdm=printdm,writedm=writedm,Nhf=Nhf,Ptype=Ptype,tobdg=tobdg,mu=mu)


