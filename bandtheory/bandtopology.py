## Band topology module

'''Band topology module: Computation of band topology'''

from math import *
import numpy as np
import cmath

import sys
sys.path.append('../lattice')
import lattice as ltc
sys.path.append('../tightbinding')
import tightbinding as tb
import bandtheory as bdth




'''Berry curvature and Chern number'''


def berrycurv(k,H,dks):
    '''
    List all of the lattice sites in a unit cell
    '''
    kcts=[k+dks[0],k-dks[2],k+dks[1],k-dks[0],k+dks[2],k-dks[1]]
    Nkcts=len(kcts)
    Hs=[H(kcts[nkct]) for nkct in range(Nkcts)]
    us=[np.linalg.eigh(Hs[nkct])[1].transpose() for nkct in range(Nkcts)]
    dus=[[np.vdot(us[(nkct+1)%Nkcts][nub],us[nkct][nub]) for nub in range(us[nkct].shape[0])] for nkct in range(Nkcts)]
    dups=[[dus[nkct][nub]/abs(dus[nkct][nub]) for nub in range(len(dus[nkct]))] for nkct in range(Nkcts)]
    deBs=list(np.prod(np.array(dups),axis=0))
    dBs=[cmath.phase(deBs[nub]) for nub in range(len(deBs))]
    Bs=[dBs[nub]/(6.*(sqrt(3)/4.)*np.linalg.norm(dks[0])**2) for nub in range(len(dBs))]
    return [dBs,Bs]
