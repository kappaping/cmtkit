## Tight-binding module

'''Tight-binding module: Setup of tight-binding models'''

from math import *
import numpy as np

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


def termmat(Mt,mt,r0,fl0,r1,fl1,Nall):
    '''
    Assign matrix elements: Assign the coupling mt between states (r0,fl0) and (r1,fl1) to the matrix Mt under Hermitian condition
    r=[nr,sl]: Lattice site at Bravais lattice site nr and sublattice sl
    fl: Flavor index
    Nall=[Nbl,Nsl,Nfl]: Bravais lattice dimension, sublattice number, flavor number
    '''
    Mt[stid(r0,fl0,Nall),stid(r1,fl1,Nall)]+=mt
    Mt[stid(r1,fl1,Nall),stid(r0,fl0,Nall)]+=np.conj(mt)


'''Set Hamiltonian'''


def tbham(htb,rs,Nall,bc,ltype):
    '''
    Tight-binding Hamiltonian: Assign the couplings htb=[v0,-t1,-t2] to the Hamiltonian H.
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


def hamsite(H,vs,rs,Nall):
    '''
    Onsite potentials: Assign the couplings vs=[v[r0],v[r1],....] to the Hamiltonian H.
    v[r]=[v0[r],v1[r],v2[r],v3[r]]: Set the potential v0[r]*sigma_0+v1[r]*sigma_1+v2[r]*sigma_2+v3[r]*sigma_3 to the site r
    The factor 1/2 is to cancel double counting from the Hermitian assignment in termmat
    '''
    for r in rs:
        # Onsite potential
        vr=vs[ltc.rid(r,Nall[0])]
        vrm=(1./2.)*(vr[0]*paulimat(0)+vr[1]*paulimat(1)+vr[2]*paulimat(2)+vr[3]*paulimat(3))
        # Add matrix elements for the pairs
        [termmat(H,(1./2.)*vrm[fl0,fl1],r,fl0,r,fl1,Nall) for fl0 in range(Nall[1]) for fl1 in range(Nall[1])]


'''Functions of density matrix'''


def projdenmat(U,n0,n1,Nst):
    '''
    Generate the density matrix by projecting on the n0-th to n1-th states of the unitary operator U=[u0,u1,u2,....].
    '''
    UT=U.conj().T
    # Project to only the Noc occupied states
    D=np.diag(np.array(n0*[0.]+(n1-n0)*[1.]+(Nst-n1)*[0.]))
    return np.linalg.multi_dot([U,D,UT])


def pairdenmat(P,r0,r1,Nall):
    '''
    Generate the 2x2 density matrix of a pair of lattice sites.
    '''
    return np.array([[P[stid(r0,fl0,Nall),stid(r1,fl1,Nall)] for fl1 in range(Nall[1])] for fl0 in range(Nall[1])])
        

def paircharge(Pt,r0,r1,Nall):
    '''
    Compute the charge of a pair of lattice sites. The onsite charge is real, while the offsite charge can be complex.
    '''
    return np.trace(pairdenmat(Pt,r0,r1,Nall))


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


def setpairspin(Pt,Pr,r0,r1,Nall):
    '''
    Set the spin of a pair of lattice sites.
    '''
    [termmat(Pt,(1./2.)*Pr[fl0,fl1],r0,fl0,r1,fl1,Nall) for fl0 in range(Nall[1]) for fl1 in range(Nall[1])]



