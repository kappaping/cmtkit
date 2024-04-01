## Kagome-lattice density matrix module

'''Kagome-lattice density matrix module'''

from math import *
import cmath as cmt
import numpy as np
from scipy.spatial.transform import Rotation
import joblib

import sys
sys.path.append('../lattice')
import lattice as ltc
import tightbinding as tb




'''Matrix setup'''


def denmatans(P,Ptype,rs,NB,RDV,Nrfl,dpe,tobdg):
    '''
    Ansatze of the density matrices on the kagome lattice.
    '''
    ltype='ka'
    [Nr,Nfl]=Nrfl
    cp120ss=[
            [[0.43300446+0.j,0.0407828+0.07454883j],
                [0.0407828-0.07454883j,0.90032935+0.j]],
            [[0.71010344+2.08166817e-17j,-0.10347812-2.21865838e-01j],
                [-0.10347812+2.21865838e-01j,0.62323038+0.00000000e+00j]],
            [[0.85689283-1.38777878e-17j,0.06269533+1.47317004e-01j],
                [0.06269533-1.47317004e-01j,0.47644099+6.93889390e-18j]]
            ]
    # Plot lattice
    if(Ptype=='pllt'):
        [tb.termmat(P,(1./2.)*dpe,rid,fl,rid,fl,Nfl,tobdg=tobdg,phid0=0,phid1=0) for rid in range(ltc.slnum(ltype)) for fl in range(Nfl)]
    # Charge nematicity
    if(Ptype=='cn'):
        [tb.termmat(P,(1./2.)*dpe*np.sign((rs[rid][1]+1)%3),rid,fl,rid,fl,Nfl,tobdg=tobdg,phid0=0,phid1=0) for rid in range(Nr) for fl in range(Nfl)]
    # Quantum anomalous Hall insulator
    elif(Ptype=='qahi'):
        nb1ids=ltc.nthneighbors(1,NB)
        # Add ccurrents
        [tb.termmat(P,(1./2.)*dpe*1.j*((rs[nb1id[0]][1]-rs[nb1id[1]][1])%3-1.5),nb1id[0],fl,nb1id[1],fl,Nfl,tobdg=tobdg,phid0=0,phid1=0) for nb1id in nb1ids for fl in range(Nfl)]
    # Ferromagnetism
    elif(Ptype=='fm'):
        [tb.termmat(P,(1./2.)*dpe*(0.5-fl),rid,fl,rid,fl,Nfl,tobdg=tobdg,phid0=0,phid1=0) for rid in range(Nr) for fl in range(Nfl)]
    # p-wave sc
    elif(Ptype=='psc'):
        ds=[np.dot(Rotation.from_rotvec((2*pi)*(nd/6)*np.array([0,0,1])).as_matrix(),ltc.blvecs(ltype)[0]) for nd in range(6)]
        pds=[cos(1*(2*pi)*(nd/6)) for nd in range(6)]
        nb1ids=ltc.nthneighbors(1,NB)
        # Add pairings
        [tb.termmat(P,(1./2.)*dpe*pds[np.argmax([np.dot(ds[nd],RDV[nb1id[1],nb1id[0]]) for nd in range(6)])],nb1id[0],fl,nb1id[1],fl,Nfl,tobdg=tobdg,phid0=0,phid1=1) for nb1id in nb1ids for fl in range(Nfl)]
    # d-wave sc
    elif(Ptype=='dsc'):
        ds=[np.dot(Rotation.from_rotvec((2*pi)*(nd/6)*np.array([0,0,1])).as_matrix(),ltc.blvecs(ltype)[0]) for nd in range(6)]
        pds=[cos(2*(2*pi)*(nd/6)) for nd in range(6)]
        nb1ids=ltc.nthneighbors(1,NB)
        # Add pairings
        [tb.termmat(P,(1./2.)*dpe*pds[np.argmax([np.dot(ds[nd],RDV[nb1id[1],nb1id[0]]) for nd in range(6)])],nb1id[0],fl,nb1id[1],fl,Nfl,tobdg=tobdg,phid0=0,phid1=1) for nb1id in nb1ids for fl in range(Nfl)]
    # f-wave sc
    elif(Ptype=='fsc'):
        ds=[np.dot(Rotation.from_rotvec((2*pi)*(nd/6)*np.array([0,0,1])).as_matrix(),ltc.blvecs(ltype)[0]) for nd in range(6)]
        pds=[cos(3*(2*pi)*(nd/6)) for nd in range(6)]
        nb1ids=ltc.nthneighbors(1,NB)
        # Add pairings
        [tb.termmat(P,(1./2.)*dpe*pds[np.argmax([np.dot(ds[nd],RDV[nb1id[1],nb1id[0]]) for nd in range(6)])],nb1id[0],fl,nb1id[1],fl,Nfl,tobdg=tobdg,phid0=0,phid1=1) for nb1id in nb1ids for fl in range(Nfl)]







