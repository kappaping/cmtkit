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


ltype='ka'
print(ltc.ltcname(ltype))
#Nbl=[6,6,1]
Nbl=[12,12,1]
Nsl=ltc.slnum(ltype)
Nltc=[Nbl,Nsl]
Nfl=2
Nall=[Nltc,Nfl]
Nst=tb.stnum(Nall)
print('State number = ',Nst)
bc=1
rs=ltc.ltcsites(Nall[0])

nfb=-1./1.
nf=5./6.+nfb*(1./6.)+(0.)/Nst
print('Flat-band filling = ',nfb,', filling = ',nf)

filet='../../data/test/test00'
#filet='data/square/afm_t2_00_u0_200_u1_00_u2_00_8x8'
#filet='../../data/hartreefock/kagome/23filling/sdwcbo/t2_00_u0_40_u1_00_u2_00_6x6'
P=joblib.load(filet)
#P=dm.setdenmat(ltype,'rand',rs,Nall,Nst,nf,bc)


print('charge order')
chs=tb.chargeorder(P,rs,Nall,ltype)
print('site order max = ',chs[1][0],', site order average = ',sum(chs[0][0])/len(chs[0][0]))
if(chs[1][0]>1e-5):print('site order = \n',chs[0][0])
print('real bond order max = ',chs[1][1],', real bond order average = ',sum(chs[0][1])/len(chs[0][1]))
#if(chs[1][1]>1e-5):print('real bond order = \n',chs[0][1])
print('imaginary bond order max = ',chs[1][2],', imaginary bond order average = ',sum(chs[0][2])/len(chs[0][2]))
#if(chs[1][2]>1e-5):print('imaginary bond order = \n',chs[0][2])

print('spin order')
sps=tb.spinorder(P,rs,Nall,ltype)
print('site order max = ',sps[1][0],', site order average = ',sum(sps[0][0])/len(sps[0][0]))
if(sps[1][0]>1e-5):
#    print('site order = \n',sps[0][0])
    print('site order norm = ')
    spns=[np.linalg.norm(sp) for sp in sps[0][0]]
#    print(spns)
print('real bond order max = ',sps[1][1],', real bond order average = ',sum(sps[0][1])/len(sps[0][1]))
#if(sps[1][1]>1e-5):print('real bond order = \n',sps[0][1])
print('imaginary bond order max = ',sps[1][2],', imaginary bond order average = ',sum(sps[0][2])/len(sps[0][2]))
#if(sps[1][2]>1e-5):print('imaginary bond order = \n',sps[0][2])


