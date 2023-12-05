## Tight-binding module

'''Tight-binding module: Setup of tight-binding models'''

from math import *
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size']=18
plt.rcParams.update({'figure.autolayout': True})

import sys
sys.path.append('../lattice')
import lattice as ltc




'''Matrix setup'''


def stateid(rid,fl,Nfl):
    '''
    Matrix indices for the fermion with fl at site r
    rid: Index of lattice site
    fl: Flavor index
    Nfl: Flavor number
    '''
    return Nfl*rid+fl


def statenum(Nrfl):
    '''
    State number
    '''
    return Nrfl[0]*Nrfl[1]


def termmat(Mt,mt,rid0,fl0,rid1,fl1,Nfl):
    '''
    Assign matrix elements: Assign the coupling mt between states (r0,fl0) and (r1,fl1) to the matrix Mt under Hermitian condition
    r=[nr,sl]: Lattice site at Bravais lattice site nr and sublattice sl
    fl: Flavor index
    Nall=[Nbl,Nsl,Nfl]: Bravais lattice dimension, sublattice number, flavor number
    '''
    Mt[stateid(rid0,fl0,Nfl),stateid(rid1,fl1,Nfl)]+=mt
    Mt[stateid(rid1,fl1,Nfl),stateid(rid0,fl0,Nfl)]+=np.conj(mt)


'''Set Hamiltonian'''


def tbham(ts,NB,Nfl):
    '''
    Tight-binding Hamiltonian: Assign the hoppings ts=[-t0,-t1,-t2,....] to the Hamiltonian H.
    '''
    print('Tight-binding model: [-t0,-t1,-t2,....] =',ts)
    # Construct a list of hoppings tnbs=[ts,0,0,....] with the length matching the number of all neighboring distances.
    maxnb=np.max(NB)
    tnbs=ts+[0. for n in range(maxnb-(len(ts)-1))]
    # Construct the tight-binding Hamiltonian with the hoppings assigned by the neighboring distances.
    H=np.array([[tnbs[nb]*(fl0==fl1) for nb in row for fl1 in range(Nfl)] for row in NB for fl0 in range(Nfl)],dtype=complex)
    return H


def sitedenimb(H,t0,Nrfl):
    '''
    Site-density imbalance
    '''
    [setpairpm(H,[0.,0.,0.,t0],rid,rid,Nrfl[1]) for rid in range(Nrfl[0])]


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


def somat(nor,nsp):
    '''
    Matrices for the spin-orbit coupling: (nor,nsp) determines the representation tau^nor sigma^nsp
    '''
    return np.kron(paulimat(nor),paulimat(nsp))


def pairmat(M,rid0,rid1,Nfl):
    '''
    Generate the Nfl x Nfl matrix of a pair of lattice sites with indices rid0 and rid1.
    '''
    return np.array([[M[stateid(rid0,fl0,Nfl),stateid(rid1,fl1,Nfl)] for fl1 in range(Nfl)] for fl0 in range(Nfl)])
        

def setpair(M,M01,rid0,rid1,Nfl):
    '''
    Set the matrix for a pair of lattice sites.
    M01: Matrix of the pairs rid0 and rid1.
    1/2 factors: Compensate with the Hermitian assignment in termmat.
    '''
    [termmat(M,(1./2.)*M01[fl0,fl1],rid0,fl0,rid1,fl1,Nfl) for fl0 in range(Nfl) for fl1 in range(Nfl)]


def setpairpm(M,v,rid0,rid1,Nfl):
    '''
    Set the matrix for a pair of lattice sites with Pauli matrices.
    v=[v0,v1,v2,v3]: Set the matrix v0*sigma_0+v1*sigma_1+v2*sigma_2+v3*sigma_3 to the pairs rid0 and rid1.
    1/2 factors: Compensate with the Hermitian assignment in termmat.
    '''
    V=v[0]*paulimat(0)+v[1]*paulimat(1)+v[2]*paulimat(2)+v[3]*paulimat(3)
    [termmat(M,(1./2.)*V[fl0,fl1],rid0,fl0,rid1,fl1,Nfl) for fl0 in range(Nfl) for fl1 in range(Nfl)]




'''Plot the energy spectrum'''

def plotenergy(H,Nrfl,nf,toprint=False,filetfig=''):
    '''
    Plot the energy spectrum of the Hamiltonian H.
    '''
    es=np.linalg.eigvalsh(H)
    Nst=statenum(Nrfl)
    Noc=round(Nst*nf)
    plt.rcParams.update({'font.size':30})
    cs=Noc*['g']+(Nst-Noc)*['b']
    plt.scatter(range(len(es)),es,c=cs)
    plt.xlabel('n')
    plt.ylabel('$e_n$')
    plt.gcf()
    if(toprint):plt.savefig(filetfig,dpi=2000,bbox_inches='tight',pad_inches=0)
    plt.show()


