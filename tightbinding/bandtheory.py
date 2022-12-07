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


def ucsite(nr,bc):
    '''
    Map the original Bravais lattice sites to the unit cell
    nr: Bravais lattice site index
    bc: Boundary condition
    '''
    dictt={
    0:nr,   # None
    21:np.array([nr[0]%2,0,0]), # 2 x 1 x 1 unit cell
    22:np.array([nr[0]%2,nr[1]%2,0]),   # 2 x 2 x 1 unit cell
    33:np.array({0:[0,0,0],1:[1,0,0],2:[0,1,0]}[nr[0]-nr[1]]),  # sqrt3 x sqrt3 x 1 unit cell
    }
    return dictt[bc]


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


def termmat(Mt,mt,r0,fl0,r1,fl1,Nall):
    '''
    Assign matrix elements: Assign the coupling mt between states (r0,fl0) and (r1,fl1) to the matrix Mt under Hermitian condition
    r=[nr,sl]: Lattice site at Bravais lattice site nr and sublattice sl
    fl: Flavor index
    Nall=[Nbl,Nsl,Nfl]: Bravais lattice dimension, sublattice number, flavor number
    '''
    Mt[stid(r0,fl0,Nall),stid(r1,fl1,Nall)]+=mt
    Mt[stid(r1,fl1,Nall),stid(r0,fl0,Nall)]+=np.conj(mt)


def tbham(htb,rs,Nall,bc,ltype):
    '''
    Tight-binding model: Assign the couplings htb=[v0,-t1,-t2] to the Hamiltonian Mt.
    v0: Onsite potential
    t1 and t2: Nearest and second neighbor hoppings
    The factor 1/2 is to cancel double counting from the Hermitian assignment in termmat
    '''
    H=np.zeros((stnum(Nall),stnum(Nall)),dtype=complex)
    for r in rs:
        # Pairs at Bravais lattice site bls
        pairst=ltc.pairs(r,Nall[0][0],bc,ltype)
        # Add matrix elements for the pairs
        [termmat(H,(1./2.)*htb[nd],pairt[0],fl,pairt[1],fl,Nall) for nd in range(len(pairst)) for pairt in pairst[nd] for fl in range(Nall[1])]
    return H


'''Density matrix and the evaluation of charge and spin orders'''


def projdenmat(U,n0,n1,Nst):
    '''
    Generate the density matrix by projecting on the n0-th to n1-th states.
    '''
    UT=U.conj().T
    # Project to only the Noc occupied states
    D=np.diag(np.array(n0*[0.]+(n1-n0)*[1.]+(Nst-n1)*[0.]))
    return np.linalg.multi_dot([U,D,UT])


def pairdenmat(Pt,r0,r1,Nall):
    '''
    Generate the 2x2 density matrix of a pair of lattice sites.
    '''
    return np.array([[Pt[stid(r0,fl0,Nall),stid(r1,fl1,Nall)] for fl1 in range(Nall[1])] for fl0 in range(Nall[1])])
        

def paircharge(Pt,r0,r1,Nall):
    '''
    Compute the charge of a pair of lattice sites. The onsite charge is real, while the offsite charge can be complex.
    '''
    return np.trace(pairdenmat(Pt,r0,r1,Nall))


def paulimat(n):
    '''
    Pauli matrices
    '''
    if(n==0):
        return np.array([[1.,0.],[0.,1.]])
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
    return np.array([np.trace(np.dot(pairdenmat(Pt,r0,r1,Nall),(1./2.)*paulimat(n))) for n in [1,2,3]])


def chargeorder(Pt,rs,Nall,ltype):
    '''
    Compute the charge order of the whole lattice. Return the lists of the site and bond orders and their maximal values.
    '''
    # Site order
    schs=[paircharge(Pt,r,r,Nall).real for r in rs]
    # Extract the order as the deviation from the average
    schsavg=sum(schs)/len(schs)
    schs=[schs[nr]-schsavg for nr in range(len(schs))]
    schsa=[abs(schs[nr]) for nr in range(len(schs))]
    schsmax=max(schsa)
    # Bond order
    bc=1
    bchs=[paircharge(Pt,pairt[0],pairt[1],Nall) for r in rs for pairt in ltc.pairs(r,Nall[0][0],bc,ltype)[1]]
    # Extract the order as the deviation from the average
    bchsavg=sum(bchs)/len(bchs)
    bchs=[bchs[nr]-bchsavg for nr in range(len(bchs))]
    # Distinguish the real and imaginary bonds
    bchsr=[bchs[nb].real for nb in range(len(bchs))]
    bchsra=[abs(bchsr[nb]) for nb in range(len(bchsr))]
    bchsi=[bchs[nb].imag for nb in range(len(bchs))]
    bchsia=[abs(bchsi[nb]) for nb in range(len(bchsi))]
    bchsrmax,bchsimax=max(bchsra),max(bchsia)
    return [[schs,bchsr,bchsi],[schsmax,bchsrmax,bchsimax]]


def spinorder(Pt,rs,Nall,ltype):
    '''
    Compute the spin order of the whole lattice. Return the lists of the site and bond orders and their maximal values.
    '''
    # Site order
    ssps=[pairspin(Pt,r,r,Nall).real for r in rs]
    sspsn=[np.linalg.norm(ssps[nr]) for nr in range(len(ssps))]
    sspsmax=max(sspsn)
    # Bond order
    bc=1
    bsps=[pairspin(Pt,pairt[0],pairt[1],Nall) for r in rs for pairt in ltc.pairs(r,Nall[0][0],bc,ltype)[1]]
    # Distinguish the real and imaginary bonds
    bspsr=[bsps[nb].real for nb in range(len(bsps))]
    bspsi=[bsps[nb].imag for nb in range(len(bsps))]
    bspsrn=[np.linalg.norm(bspsr[nb]) for nb in range(len(bspsr))]
    bspsin=[np.linalg.norm(bspsi[nb]) for nb in range(len(bspsi))]
    bspsrmax,bspsimax=max(bspsrn),max(bspsin)
    return [[ssps,bspsr,bspsi],[sspsmax,bspsrmax,bspsimax]]







