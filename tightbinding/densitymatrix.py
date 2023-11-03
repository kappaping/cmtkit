## Density-matrix module

'''Density-matrix module: Manipulations of density matrix.'''

from math import *
import numpy as np

import sys
sys.path.append('../lattice')
import lattice as ltc
import tightbinding as tb




def projdenmat(U,n0,n1,Nst):
    '''
    Generate the density matrix by projecting on the n0-th to n1-th states of the unitary operator U=[u0,u1,u2,....].
    '''
    UT=U.conj().T
    # Project to only the Noc occupied states
    D=np.diag(np.array(n0*[0.]+(n1-n0)*[1.]+(Nst-n1)*[0.]))
    return np.round(np.linalg.multi_dot([U,D,UT]),25)


def paircharge(P,rid0,rid1,Nfl):
    '''
    Compute the charge of a pair of lattice sites. The onsite charge is real, while the offsite charge can be complex.
    '''
    return np.trace(tb.pairmat(P,rid0,rid1,Nfl))


def pairspin(P,rid0,rid1,Nfl):
    '''
    Compute the spin of a pair of lattice sites. The onsite spin is real, while the offsite spin can be complex.
    '''
    if(Nfl==2):smats=[tb.paulimat(n+1) for n in range(3)]
    elif(Nfl==4):smats=[tb.somat(0,n+1) for n in range(3)]
#    elif(Nfl==4):smats=[tb.somat(n+1,0) for n in range(3)]
    return np.array([np.trace(np.dot(tb.pairmat(P,rid0,rid1,Nfl),(1./2.)*smats[n])) for n in range(3)])


def pairorbital(P,rid0,rid1,Nfl):
    '''
    Compute the orbital of a pair of lattice sites. The onsite orbital is real, while the offsite orbital can be complex.
    '''
    if(Nfl==4):omats=[tb.somat(n+1,0) for n in range(3)]
    return np.array([np.trace(np.dot(tb.pairmat(P,rid0,rid1,Nfl),(1./2.)*omats[n])) for n in range(3)])


def chargeorder(P,nb1ids,Nrfl):
    '''
    Compute the charge order of the whole lattice. Return the lists of the site and bond orders and their maximal values.
    '''
    # Site order
    schs=[paircharge(P,rid,rid,Nrfl[1]).real for rid in range(Nrfl[0])]
    # Extract the order as the deviation from the average
    schsavg=sum(schs)/len(schs)
    schs=[schs[nr]-schsavg for nr in range(len(schs))]
    schsa=[abs(schs[nr]) for nr in range(len(schs))]
    schsmax=max(schsa)
    # Bond order
    bchs=[paircharge(P,pair[0],pair[1],Nrfl[1]) for pair in nb1ids]
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


def spinorder(P,nb1ids,Nrfl):
    '''
    Compute the spin order of the whole lattice. Return the lists of the site and bond orders and their maximal values.
    '''
    # Site order
    ssps=[pairspin(P,rid,rid,Nrfl[1]).real for rid in range(Nrfl[0])]
    sspsn=[np.linalg.norm(ssps[nr]) for nr in range(len(ssps))]
    sspsmax=max(sspsn)
    # Bond order
    bsps=[pairspin(P,pair[0],pair[1],Nrfl[1]) for pair in nb1ids]
    # Distinguish the real and imaginary bonds
    bspsr=[bsps[nb].real for nb in range(len(bsps))]
    bspsi=[bsps[nb].imag for nb in range(len(bsps))]
    bspsrn=[np.linalg.norm(bspsr[nb]) for nb in range(len(bspsr))]
    bspsin=[np.linalg.norm(bspsi[nb]) for nb in range(len(bspsi))]
    bspsrmax,bspsimax=max(bspsrn),max(bspsin)

    return [[ssps,bspsr,bspsi],[sspsmax,bspsrmax,bspsimax]]


def orbitalorder(P,nb1ids,Nrfl):
    '''
    Compute the orbital order of the whole lattice. Return the lists of the site and bond orders and their maximal values.
    '''
    # Site order
    sobs=[pairorbital(P,rid,rid,Nrfl[1]).real for rid in range(Nrfl[0])]
    sobsn=[np.linalg.norm(sobs[nr]) for nr in range(len(sobs))]
    sobsmax=max(sobsn)
    # Bond order
    bobs=[pairorbital(P,pair[0],pair[1],Nrfl[1]) for pair in nb1ids]
    # Distinguish the real and imaginary bonds
    bobsr=[bobs[nb].real for nb in range(len(bobs))]
    bobsi=[bobs[nb].imag for nb in range(len(bobs))]
    bobsrn=[np.linalg.norm(bobsr[nb]) for nb in range(len(bobsr))]
    bobsin=[np.linalg.norm(bobsi[nb]) for nb in range(len(bobsi))]
    bobsrmax,bobsimax=max(bobsrn),max(bobsin)

    return [[sobs,bobsr,bobsi],[sobsmax,bobsrmax,bobsimax]]







