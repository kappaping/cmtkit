## General lattice density matrix module

'''General lattice density matrix module'''

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
    Ptypet=Ptype[3:]
    [Nr,Nfl]=Nrfl
    # Plot lattice
    if(Ptypet=='pllt'):
        [tb.termmat(P,(1./2.)*dpe,rid,fl,rid,fl,Nfl,tobdg=tobdg,phid0=0,phid1=0) for rid in range(ltc.slnum(ltype)) for fl in range(Nfl)]
    # Charge-demsity modulation
    if(Ptypet=='cdm0'):
        [tb.termmat(P,(1./2.)*dpe*(0.5-rs[rid][1]),rid,fl,rid,fl,Nfl,tobdg=tobdg,phid0=0,phid1=0) for rid in range(Nr) for fl in range(Nfl)]
    if(Ptypet=='cdm1'):
        [tb.termmat(P,(1./2.)*dpe*np.sign(0.5-rs[rid][1]),rid,fl,rid,fl,Nfl,tobdg=tobdg,phid0=0,phid1=0) for rid in range(Nr) for fl in range(Nfl)]
    # Ferromagnetism
    elif(Ptypet=='fm'):
        [tb.termmat(P,(1./2.)*dpe*(0.5-fl),rid,fl,rid,fl,Nfl,tobdg=tobdg,phid0=0,phid1=0) for rid in range(Nr) for fl in range(Nfl)]
    # Antiferromagnetism
    elif(Ptypet=='afm'):
        [tb.termmat(P,(1./2.)*dpe*(0.5-fl)*(0.5-rs[rid][1]),rid,fl,rid,fl,Nfl,tobdg=tobdg,phid0=0,phid1=0) for rid in range(Nr) for fl in range(Nfl)]
    # Antiferromagnetism2d
    elif(Ptypet=='afm2d'):
        [tb.setpairpm(P,dpe*((-1)**rs[rid][1])*np.array([0,np.sin(pi/5)*np.cos(7*pi/6),np.sin(pi/5)*np.sin(7*pi/6),np.cos(pi/5)]),rid,rid,Nfl) for rid in range(Nr) for fl in range(Nfl)]
    # Antiferromagnetism2d
    elif(Ptypet=='afm3d'):
        [tb.setpairpm(P,dpe*((-1)**rs[rid][1])*np.array([0,np.sin(pi/5)*np.cos(7*pi/6),np.sin(pi/5)*np.sin(7*pi/6),np.cos(pi/5)]),rid,rid,Nfl) for rid in range(Nr) for fl in range(Nfl)]
    # Ferromagnetism + charge-density modulation
    elif(Ptypet=='fmcdm0'):
        [tb.termmat(P,(1./2.)*dpe*((0.5-fl)+(0.5-rs[rid][1])),rid,fl,rid,fl,Nfl,tobdg=tobdg,phid0=0,phid1=0) for rid in range(Nr) for fl in range(Nfl)]
    elif(Ptypet=='fmcdm1'):
        [tb.termmat(P,(1./2.)*dpe*((0.5-fl)+np.sign(0.5-rs[rid][1])),rid,fl,rid,fl,Nfl,tobdg=tobdg,phid0=0,phid1=0) for rid in range(Nr) for fl in range(Nfl)]







