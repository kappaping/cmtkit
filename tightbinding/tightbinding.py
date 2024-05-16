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
import bogoliubovdegennes as bdg




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


def termmat(M,m,rid0,fl0,rid1,fl1,Nfl,symtype='herm',tobdg=False,phid0=0,phid1=0):
    '''
    Assign matrix elements: Assign the coupling mt between states (rid0,fl0) and (rid1,fl1) to the matrix Mt under Hermitian condition.
    M: Matrix to be modified.
    m: Matrix element to be added.
    rid0, rid1: Lattice site indices.
    fl0, fl1: Flavor indices.
    Nfl: Flavor number.
    '''
    if(symtype=='herm'):mt=np.conj(m)
    elif(symtype=='symm'):mt=m
    elif(symtype=='asym'):mt=-m
    if(tobdg==False):
        M[stateid(rid0,fl0,Nfl),stateid(rid1,fl1,Nfl)]+=m
        M[stateid(rid1,fl1,Nfl),stateid(rid0,fl0,Nfl)]+=mt
    elif(tobdg):
        Nst=round(M.shape[0]/2)
        M[phid0*Nst+stateid(rid0,fl0,Nfl),phid1*Nst+stateid(rid1,fl1,Nfl)]+=m
        M[phid1*Nst+stateid(rid1,fl1,Nfl),phid0*Nst+stateid(rid0,fl0,Nfl)]+=mt


'''Set Hamiltonian'''


def tbham(ts,NB,Nfl):
    '''
    Tight-binding Hamiltonian: Assign the hoppings ts=[-t0,-t1,-t2,....] to the Hamiltonian H.
    '''
    # Construct the tight-binding Hamiltonian with the hoppings assigned by the neighboring distances.
    H=np.zeros((Nfl*(NB.shape[0]),Nfl*(NB.shape[1])),dtype=complex)
    tfs=[]
    issimtb=True
    for nt in range(len(ts)):
        tf=ts[nt]
        if(callable(ts[nt])):issimtb=False
        else:
            def tf(rid0,rid1):return ts[nt]*np.identity(Nfl)
        nbs=np.argwhere(NB==nt)
        for nb in nbs:
            setpair(H,tf(nb[0],nb[1]),nb[0],nb[1],Nfl)
    if(issimtb):print('Tight-binding model: [-t0,-t1,-t2,....] =',ts,', flavor number =',Nfl)
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


def pairmat(M,rid0,rid1,Nfl,tobdg=False,phid0=0,phid1=0):
    '''
    Get the Nfl x Nfl matrix of a pair of lattice sites with indices rid0 and rid1.
    '''
    Mt=M
    if(tobdg):Mt=bdg.bdgblock(M,phid0,phid1)
    return np.array([[Mt[stateid(rid0,fl0,Nfl),stateid(rid1,fl1,Nfl)] for fl1 in range(Nfl)] for fl0 in range(Nfl)])
        

def setpair(M,M01,rid0,rid1,Nfl,tobdg=False,phid0=0,phid1=0):
    '''
    Set the matrix for a pair of lattice sites.
    M01: Matrix of the pairs rid0 and rid1.
    1/2 factors: Compensate with the Hermitian assignment in termmat.
    '''
    [termmat(M,(1./2.)*M01[fl0,fl1],rid0,fl0,rid1,fl1,Nfl,tobdg=tobdg,phid0=phid0,phid1=phid1) for fl0 in range(Nfl) for fl1 in range(Nfl)]


def setpairpm(M,v,rid0,rid1,Nfl):
    '''
    Set the matrix for a pair of lattice sites with Pauli matrices.
    v=[v0,v1,v2,v3]: Set the matrix v0*sigma_0+v1*sigma_1+v2*sigma_2+v3*sigma_3 to the pairs rid0 and rid1.
    1/2 factors: Compensate with the Hermitian assignment in termmat.
    '''
    V=v[0]*paulimat(0)+v[1]*paulimat(1)+v[2]*paulimat(2)+v[3]*paulimat(3)
    [termmat(M,(1./2.)*V[fl0,fl1],rid0,fl0,rid1,fl1,Nfl) for fl0 in range(Nfl) for fl1 in range(Nfl)]




'''Temperature.'''


def fermi(z,T):
    '''
    Define the Fermi function.
    Return: Fermi function 1/(e^(z/T)+1).
    z: Characteristic energy, usually set as ee-mu with energy ee and chemical potential mu.
    T: Temperature.
    '''
    if(abs(z/T)<=32.):return 1./(e**(z/T)+1.)
    elif(z/T>32.):return 0.
    elif(z/T<-32.):return 1.




'''Plot the energy spectrum'''

def plotenergy(H,Nrfl,nf,toprint=False,filetfig='',tobdg=False):
    '''
    Plot the energy spectrum of the Hamiltonian H.
    '''
    ees=np.linalg.eigvalsh(H)
    Nst=statenum(Nrfl)
    if(tobdg==False):Noc=round(Nst*nf)
    elif(tobdg):
        Noc=Nst
        Nst*=2
    print('Chemical potential mu =',ees[Noc-1])
    plt.rcParams.update({'font.size':30})
    cs=Noc*['g']+(Nst-Noc)*['b']
    plt.scatter(range(len(ees)),ees,c=cs)
    plt.xlabel('n')
    plt.ylabel('$e_n$')
    plt.gcf()
    if(toprint):plt.savefig(filetfig,dpi=2000,bbox_inches='tight',pad_inches=0)
    plt.show()


