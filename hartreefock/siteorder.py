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
import flatband as fb


ltype='ka'
print(ltc.ltcname(ltype))
#Nbl=[6,6,1]
Nbl=[12,12,1]
#Nbl=[24,24,1]
#Nbl=[4,4,4]
#Nbl=[6,6,6]
Nsl=ltc.slnum(ltype)
Nltc=[Nbl,Nsl]
Nfl=2
Nall=[Nltc,Nfl]
Nst=tb.stnum(Nall)
print('System size = ',Nbl)
print('State number = ',Nst)
bc=1
rs=ltc.ltcsites(Nall[0])

nfb=-1./1.
nf=fb.totfillfromfb(ltype,nfb)+(0.)/Nst
print('Flat-band filling = ',nfb,', filling = ',nf)

#filet='../../data/test/test01'
filet='../../data/hartreefock/kagome/nfbn1/120sdw/t2_00_u0_40_u1_00_u2_00_12x12'
#filet='../../data/hartreefock/pyrochlore/nfbn1/tsfcc/t2_00_u0_50_u1_00_u2_00_6x6x6'
P=joblib.load(filet)
#P=dm.setdenmat(ltype,'sdwt22',rs,Nall,Nst,nf,bc)
#'''
rtls=[
        [[0,0,0],0],[[1,0,0],0],[[0,1,0],0]#,[[0,0,1],0]
#        [[0,0,0],0],[[0,0,0],1],[[0,0,0],2],[[0,0,0],3]
        ]
#'''
'''
rtls=[[r,sl] for r in [[0,0,0],[1,0,0],[0,1,0],[0,0,1]] for sl in range(4)]
'''
#rtls=[[[0,n,0],2*sl] for n in range(Nbl[1]) for sl in range(2)]
#rtls=[[[0,0,0],sl] for sl in range(Nsl)]
#rtls=[[[n0,n1,0],sl] for n0 in range(Nbl[0]) for n1 in range(Nbl[1]) for sl in range(Nsl)]
'''
# 120TAFM
rtls=[[rbls,sl] for rbls in [[0,0,0],[1,0,0],[0,1,0]] for sl in range(ltc.slnum(ltype))]
'''
'''
# TCTS
rtls=[[rbls,sl] for rbls in [[2,0,0],[3,0,0],[2,1,0],[3,1,0],[4,1,0],[3,2,0],[1,2,0],[2,2,0],[1,3,0],[2,3,0],[3,3,0],[2,4,0]] for sl in range(ltc.slnum(ltype))]
'''
'''
# THSV
rtls=[[rbls,sl] for rbls in [[2,0,0],[3,0,0],[2,1,0],[3,1,0],[4,1,0],[3,2,0],[1,2,0],[2,2,0],[1,3,0],[2,3,0],[3,3,0],[2,4,0]] for sl in range(ltc.slnum(ltype))]
'''
'''
# THSV higher density spin vortex
rtls=[
        [[2,0,0],1],[[2,0,0],2],
        [[1,2,0],1],[[1,2,0],2],
        [[3,1,0],1],[[3,1,0],2],
        [[2,3,0],1],[[2,3,0],2],
        ]
'''
rts=[[np.array(rtl[0]),rtl[1]] for rtl in rtls]
cs=[tb.paircharge(P,rt,rt,Nall).real for rt in rts]
ss=[tb.pairspin(P,rt,rt,Nall).real for rt in rts]
ps=[tb.pairdenmat(P,rt,rt,Nall) for rt in rts]

#nfb=-5/6 tts
#ss=[(ss[3*n]+ss[3*n+1]+ss[3*n+2])/3. for n in range(4)]

#nfb=-2/3 thsv
#ss=[np.cross(ss[6*n],ss[6*n+1]) for n in range(4)]
#ss=[np.cross(ss[6*n+2*m],ss[6*n+2*m+1]) for n in range(4) for m in range(3)]
#ss=[(ss[3*n]+ss[3*n+1]+ss[3*n+2])/3. for n in range(4)]

#sns=[np.linalg.norm(s) for s in ss]
#nss=[ss[n]/sns[n] for n in range(len(ss))]
[print(cs[n],',') for n in range(len(ss))]
[print('{',ss[n][0],',',ss[n][1],',',ss[n][2],'},') for n in range(len(ss))]
Ufl=np.array([[0.965926-0.12941j,-0.194114-0.112072j],[0.194114-0.112072j,0.965926+0.12941j]])
UflT=Ufl.conj().T
[print(np.linalg.multi_dot([Ufl,ps[n],UflT]),',') for n in range(len(ps))]


