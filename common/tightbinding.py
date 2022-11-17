## Tight-binding module

'''Tight-binding module: For the setup of tight-binding models'''

from math import *
import cmath as cmt
import numpy as np

import lattice as ltc




'''Matrix setup'''


def stid(r,fl,Nall):
    '''
    Matrix indices for the fermion with fl at site r
    r: Lattice site
    flavor: Flavor index
    Nall=[Nltc,Nfl]: Lattice dimension, flavor number
    '''
    return Nall[1]*ltc.rid(r,Nall[0])+fl


def termmat(Mt,mt,r1,fl1,r2,fl2,Nall):
    '''
    Assign matrix elements: Assign the coupling mt between states (r1,fl1) and (r2,fl2) to the matrix Mt under Hermitian condition
    r=[nr,sl]: Lattice site at Bravais lattice site nr and sublattice sl
    fl: Flavor index
    Nall=[Nbl,Nsl,Nfl]: Bravais lattice dimension, sublattice number, flavor number
    '''
    Mt[stid(r1,fl1,Nall),stid(r2,fl2,Nall)]+=mt
    Mt[stid(r2,fl2,Nall),stid(r1,fl1,Nall)]+=np.conj(mt)


def tbham(Mt,htb,rs,Nall,bc,mtype):
    '''
    Tight-binding model: Assign the couplings htb=[v0,-t1,-t2] to the Hamiltonian Mt.
    v0: Onsite potential
    t1 and t2: Nearest and second neighbor hoppings
    The factor 1/2 is to cancel double counting from the Hermitian assignment in termmat
    '''
    for r in rs:
        # Pairs at Bravais lattice site bls
        pairst=ltc.pairs(r,Nall[0][0],bc,mtype)
        # Add matrix elements for the pairs
        [[termmat(Mt,(1./2.)*htb[nd],pairt[0],fl,pairt[1],fl,Nall) for pairt in pairst[nd] for fl in range(Nall[1])] for nd in range(len(pairst))]



