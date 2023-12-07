## Bogoliubov-de Gennes module

'''Bogoliubov-de Gennes (BdG) module: Functions of BdG'''

from math import *
import cmath
import numpy as np

import sys
sys.path.append('../lattice')
import lattice as ltc
import tightbinding as tb




'''BdG formalism'''


def phmattobdg(M,isham=False,mu=0.):
    '''
    Construct a Bogoliubov-de Gennes matrix from the particle-hole one M.
    '''
    Nst=M.shape[0]
    Mt=np.copy(M)
    if(isham):Mt-=mu*np.identity(Nst)
    Mbdg=np.block([[Mt,np.zeros((Nst,Nst))],[np.zeros((Nst,Nst)),-Mt.T]])
    if(isham):Mbdg*=1./2.
    return Mbdg


def bdgblock(M,phid0,phid1):
    '''
    Extract the [phid0,phid1] block in the Bogoliubov-de Gennes matrix M.
    '''
    Nst=round(M.shape[0]/2)
    return M[phid0*Nst:(phid0+1)*Nst,phid1*Nst:(phid1+1)*Nst]


def denmatfilling(P,Nst):
    '''
    Compute the filling of a density matrix P.
    '''
    return np.trace(bdgblock(P,0,0)).real/Nst




'''Compute the pairing orders'''


def pairpairing(P,rid0,rid1,Nfl):
    '''
    Get the pairings of a pair of lattice sites with indices rid0 and rid1.
    '''
    return np.array([np.trace(np.dot(tb.pairmat(P,rid0,rid1,Nfl,tobdg=True,phid0=0,phid1=1),np.dot((1./sqrt(2.))*tb.paulimat(n),1.j*tb.paulimat(2)).conj().T)) for n in [0,1,2,3]])


def flavoroddorder(P,nb1ids,Nrfl):
    '''
    Compute the singlet pairing of the whole lattice. Return the lists of the site and bond orders and their maximal values.
    '''
    # Site order
    sfos=[pairpairing(P,rid,rid,Nrfl[1])[0].real for rid in range(Nrfl[0])]
    # Rotate the site orders and cancel the complex phase.
#    cphase=e**(-1.j*cmath.phase(sfos[0]))
#    sfos=[(sfo*cphase).real for sfo in sfos]
    # Extract the order as the deviation from the average
    sfosa=[abs(sfo) for sfo in sfos]
    sfosmax=max(sfosa)
    # Bond order
    bfos=[pairpairing(P,pair[0],pair[1],Nrfl[1])[0] for pair in nb1ids]
    # Distinguish the real and imaginary bonds
    bfosr=[bfo.real for bfo in bfos]
    bfosra=[abs(bfor) for bfor in bfosr]
    bfosi=[bfo.imag for bfo in bfos]
    bfosia=[abs(bfoi) for bfoi in bfosi]
    bfosrmax,bfosimax=max(bfosra),max(bfosia)

    return [[sfos,bfosr,bfosi],[sfosmax,bfosrmax,bfosimax]]


def flavorevenorder(P,nb1ids,Nrfl):
    '''
    Compute the triplet pairing of the whole lattice. Return the lists of the site and bond orders and their maximal values.
    '''
    # Site order
    sfes=[pairpairing(P,rid,rid,Nrfl[1])[1:4].real for rid in range(Nrfl[0])]
    sfesn=[np.linalg.norm(sfe) for sfe in sfes]
    sfesmax=max(sfesn)
    # Bond order
    bfes=[pairpairing(P,pair[0],pair[1],Nrfl[1])[1:4] for pair in nb1ids]
    # Distinguish the real and imaginary bonds
    bfesr=[bfe.real for bfe in bfes]
    bfesi=[bfe.imag for bfe in bfes]
    bfesrn=[np.linalg.norm(bfer) for bfer in bfesr]
    bfesin=[np.linalg.norm(bfei) for bfei in bfesi]
    bfesrmax,bfesimax=max(bfesrn),max(bfesin)

    return [[sfes,bfesr,bfesi],[sfesmax,bfesrmax,bfesimax]]




