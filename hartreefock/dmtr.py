## Pyrochlore-lattice density matrix module

'''Pyrochlore-lattice density matrix module'''

from math import *
import cmath as cmt
import numpy as np
from scipy.stats import unitary_group
import joblib

import sys
sys.path.append('../../cmt_code/lattice')
import lattice as ltc
sys.path.append('../../cmt_code/tightbinding')
import tightbinding as tb




'''Matrix setup'''


def denmatans(P,Ptype,rs,Nrfl,Nbl,bc,dpe):
    '''
    Ansatze of the density matrices.
    '''
    # Plot lattice
    if(Ptype=='pllt'):
        [tb.termmat(P,(1./2.)*dpe*(0.5-fl),rid,fl,rid,fl,Nrfl[1]) for rid in range(Nr) for fl in range(Nrfl[1])]
    # Tetrehedral spin order, sqrt2xsqrt2
    elif(Ptype=='ts22'):
        ss=0.5*np.array([[0,1,1,1],[0,1,-1,-1],[0,-1,1,-1],[0,-1,-1,1]])
        def tsid(r):
            r22=[r[0][0]%2,r[0][1]%2]
            if(r22==[0,0]):return 0
            elif(r22==[0,1]):return 1
            elif(r22==[1,0]):return 2
            elif(r22==[1,1]):return 3
        [tb.setpairpm(P,ss[tsid(rs[rid])],rid,rid,Nrfl[1]) for rid in range(Nrfl[0])]
    # Simple cubic spin order, sqrt2xsqrt2
    elif(Ptype=='scb2223'):
        def cid(r):
            rbl2=[(r[0][1]-r[0][2])%2,(r[0][2]-r[0][0])%2,(r[0][0]-r[0][1])%2]
            if(rbl2==[0,0,0]):return 1
            else:return 0
        [tb.setpair(P,cid(rs[rid])*(0.5-rs[rid][0][0]%2)*np.array([[1.,0.],[0.,-1.]]),rid,rid,Nrfl[1]) for rid in range(Nrfl[0])]
    # Simple cubic afm, 2x2x2
    elif(Ptype=='scbafm222'):
        def cid(r):
            rbl2=[(r[0][1]-r[0][2])%2,(r[0][2]-r[0][0])%2,(r[0][0]-r[0][1])%2]
            if(rbl2==[0,0,0] and r[1]==0):return 1
            else:return 0
        rsts=[[r,sl] for r in [np.array([0,0,0]),np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1])] for sl in range(4)]
        for rid in range(Nrfl[0]):
            if(cid(rs[rid])):
                rbl=rs[rid][0]
                [tb.setpair(P,(0.5-rbl[0]%2)*np.array([[1.,0.],[0.,-1.]]),ltc.siteid([ltc.cyc(rbl+rst[0],Nbl,bc),rst[1]],rs),ltc.siteid([ltc.cyc(rbl+rst[0],Nbl,bc),rst[1]],rs),Nrfl[1]) for rst in rsts]







