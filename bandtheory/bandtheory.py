## Band module

'''Band module: Setup of band theory'''

from math import *
import cmath as cmt
import numpy as np
import sympy

import sys
sys.path.append('../lattice')
import lattice as ltc
sys.path.append('../tightbinding')
import tightbinding as tb




'''Matrix setup'''


def ucsites(ltype,uctype):
    '''
    List all of the lattice sites in a unit cell
    '''
    dictt={
    111:[np.array([0,0,0])],   # 1 x 1 x 1
    211:[np.array([n0,0,0]) for n0 in range(2)], # 2 x 1 x 1
    121:[np.array([0,n1,0]) for n1 in range(2)], # 2 x 1 x 1
    221:[np.array([n0,n1,0]) for n0 in range(2) for n1 in range(2)],   # 2 x 2 x 1
    23231:[np.array([n0,n1,0]) for n0 in range(2) for n1 in range(2-n0)]  # sqrt3 x sqrt3 x 1
    }
    return [[nr,sl] for nr in dictt[uctype] for sl in range(ltc.slnum(ltype))]


def ucnums(uctype):
    '''
    Give the dimensions of the Bravais lattice unit cells
    '''
    dictt={
    111:[1,1,1],  # 1 x 1 x 1
    211:[2,1,1], # 2 x 1 x 1
    121:[1,2,1], # 2 x 1 x 1
    221:[2,2,1], # 2 x 2 x 1
    23231:[2,2,1], # sqrt3 x sqrt3 x 1
    }
    return dictt[uctype]


def ucstnum(ltype,uctype,Nfl):
    '''
    State number in the unit cell
    '''
    return len(ucsites(ltype,uctype))*Nfl


def tbham(htb,k,ltype,uctype,Nfl):
    '''
    Tight-binding model in momentum space: Assign the couplings htb=[v0,-t1,-t2] to the Hamiltonian H
    v0: Onsite potential
    t1 and t2: Nearest and second neighbor hoppings
    The factor 1/2 is to cancel double counting from the Hermitian assignment in termmat
    '''
    Nalluc=[[ucnums(uctype),ltc.slnum(ltype)],Nfl]
    rs=ucsites(ltype,uctype)
    H=np.zeros((ucstnum(ltype,uctype,Nfl),ucstnum(ltype,uctype,Nfl)),dtype=complex)
    for r in rs:
        # Pairs at Bravais lattice site r
        pairs0=ltc.pairs(r,ucnums(uctype),0,ltype)
        # Pairs in the unit cell
        pairsuc=ltc.pairs(r,ucnums(uctype),uctype,ltype)
        # Add matrix elements for the pairs
        [tb.termmat(H,(1./2.)*htb[nd]*e**(-1.j*np.dot(k,(ltc.pos(pairs0[nd][npr][0],ltype)-ltc.pos(pairs0[nd][npr][1],ltype)))),pairsuc[nd][npr][0],fl,pairsuc[nd][npr][1],fl,Nalluc) for nd in range(len(pairs0)) for npr in range(len(pairs0[nd])) for fl in range(Nfl)]
    return H









