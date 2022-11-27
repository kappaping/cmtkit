## Tight-binding module

'''Tight-binding module: Setup of tight-binding models'''

from math import *
import cmath as cmt
import numpy as np
import sympy

import sys
sys.path.append('../lattice')
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


def stnum(Nall):
    '''
    State number
    '''
    return Nall[0][0][0]*Nall[0][0][1]*Nall[0][0][2]*Nall[0][1]*Nall[1]


def termmat(Mt,mt,r1,fl1,r2,fl2,Nall):
    '''
    Assign matrix elements: Assign the coupling mt between states (r1,fl1) and (r2,fl2) to the matrix Mt under Hermitian condition
    r=[nr,sl]: Lattice site at Bravais lattice site nr and sublattice sl
    fl: Flavor index
    Nall=[Nbl,Nsl,Nfl]: Bravais lattice dimension, sublattice number, flavor number
    '''
    Mt[stid(r1,fl1,Nall),stid(r2,fl2,Nall)]+=mt
    Mt[stid(r2,fl2,Nall),stid(r1,fl1,Nall)]+=np.conj(mt)


def tbham(Mt,htb,rs,Nall,bc,ltype):
    '''
    Tight-binding model: Assign the couplings htb=[v0,-t1,-t2] to the Hamiltonian Mt.
    v0: Onsite potential
    t1 and t2: Nearest and second neighbor hoppings
    The factor 1/2 is to cancel double counting from the Hermitian assignment in termmat
    '''
    for r in rs:
        # Pairs at Bravais lattice site bls
        pairst=ltc.pairs(r,Nall[0][0],bc,ltype)
        # Add matrix elements for the pairs
        [termmat(Mt,(1./2.)*htb[nd],pairt[0],fl,pairt[1],fl,Nall) for nd in range(len(pairst)) for pairt in pairst[nd] for fl in range(Nall[1])]


'''Density matrix and the evaluation of charge and spin orders'''


def projdenmat(Ut,n0,n1,Nst):
    '''
    Generate the density matrix by projecting on the n0-th to n1-th states.
    '''
    UtT=Ut.conj().T
    # Project to only the Noc occupied states
    D=np.diag(np.array([0.]*n0+[1.]*(n1-n0)+[0.]*(Nst-n1)))
    return np.linalg.multi_dot([Ut,D,UtT])


def pairdm(Pt,r0,r1,Nall):
    '''
    Generate the 2x2 density matrix of a pair of lattice sites.
    '''
    return np.array([[Pt[stid(r0,fl0,Nall),stid(r1,fl1,Nall)] for fl1 in range(Nall[1])] for fl0 in range(Nall[1])])
        

def paircharge(Pt,r0,r1,Nall):
    '''
    Compute the charge of a pair of lattice sites. The onsite charge is real, while the offsite charge can be complex.
    '''
    pcharge=np.trace(pairdm(Pt,r0,r1,Nall))
    return pcharge.real,pcharge.imag


def paulimat(n):
    '''
    Pauli matrices
    '''
    if(n==0):
        return np.array([[0.,0.],[0.,0.]])
    elif(n==1):
        return np.array([[0.,1.],[1.,0.]])
    elif(n==2):
        return np.array([[0.,-1.j],[1.j,0.]])
    elif(n==3):
        return np.array([[1.,0.],[0.,-1.]])


def pairspin(Pt,r0,r1,Nall):
    '''
    Compute the spin of a pair of lattice sites. The onsite spin is real, while the offsite spin can be complex.
    '''
    pspin=np.array([np.trace(np.dot(pairdm(Pt,r0,r1,Nall),(1./2.)*paulimat(n))) for n in [1,2,3]])
    return pspin.real,pspin.imag








