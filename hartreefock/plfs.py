## Plot

import numpy as np
import matplotlib.pyplot as plt
import joblib
import sys
sys.path.append('../../cmt_code/lattice')
import lattice as ltc
sys.path.append('../../cmt_code/tightbinding')
import tightbinding as tb
import meanfieldham as mfh
sys.path.append('../../cmt_code/bandtheory')
import plotband as plbd
import densitymatrix as dm





ltype='ka'
print(ltc.ltcname(ltype))
Nbl=[12,12,1]
Nsl=ltc.slnum(ltype)
Nltc=[Nbl,Nsl]
Nfl=2
Nall=[Nltc,Nfl]
Nst=tb.stnum(Nall)
bc=1
rs=ltc.ltcsites(Nall[0])

nfb=-5./6.
nf=5./6.+nfb*(1./6.)+(0.)/Nst
print('Flat-band filling = ',nfb,', filling = ',nf)

htb=[0.,-1.,0.]
uhs=[6.,0.,0.]
mu=0.
uctype=23231
Nk=48

#filet='../../data/test3'
filet='../../data/hartreefock/kagome/nfbn1/120sdw/t2_00_u0_60_u1_00_u2_00_12x12'
P=joblib.load(filet)
#P=dm.setdenmat(ltype,'fmqahi',rs,Nall,Nst,nf,bc)

H=lambda k:mfh.meanfieldham(k,P,htb,uhs,Nall,ltype,uctype)

todata=True

tosetde=True
de=1./(Nk**1.5)
dataks=plbd.fermisurface(H,nf,ltype,uctype,Nk,tosetde,de)

filetfig='../../figs/bz.pdf'
tosave=True
plbd.plotbz(ltype,uctype,todata,dataks)#,tosave,filetfig)






