## Lattice module

'''Lattice module: For the setup of lattices'''

from math import *
import numpy as np

import square
import kagome
import pyrochlore




'''Lattice structure'''


def ltcname(ltype):
    '''
    Lattice name
    '''
    if(ltype==221):return square.ltcname()
    elif(ltype==233):return kagome.ltcname()
    elif(ltype==334):return pyrochlore.ltcname()


def avs(ltype):
    '''
    Bravais lattice vectors
    '''
    if(ltype==221):return square.avs()
    elif(ltype==233):return kagome.avs()
    elif(ltype==334):return pyrochlore.avs()


def ltcsites(Nltc):
    '''
    List all of the lattice sites
    '''
    return [[np.array([n0,n1,n2]),sl] for n0 in range(Nltc[0][0]) for n1 in range(Nltc[0][1]) for n2 in range(Nltc[0][2]) for sl in range(Nltc[1])]


def avsls(ltype):
    '''
    Sublattice vectors
    '''
    if(ltype==221):return square.avsls()
    elif(ltype==233):return kagome.avsls()
    elif(ltype==334):return pyrochlore.avsls()


def nslf(ltype):
    '''
    Sublattice number
    '''
    return np.shape(avsls(ltype))[0]


def rid(r,Nltc):
    '''
    Indices for the lattice site r
    r=[nr,sl]: Lattice site at Bravais lattice site nr and sublattice sl
    Nltc=[Nbl,Nsl]: Bravais lattice dimension, sublattice number
    '''
    return Nltc[1]*(Nltc[0][2]*(Nltc[0][1]*r[0][0]+r[0][1])+r[0][2])+r[1]


def pos(r,ltype):
    '''
    Site position
    r=[nr,sl]: Lattice site index
    nr: Bravais lattice site index
    sls: Sublattice site index
    '''
    return np.dot(np.array(r[0]),avs(ltype))+avsls(ltype)[r[1]]


def cyc(nr,Nbl,bc):
    '''
    Cyclic site index for periodic boundary condition (PBC)
    nr: Bravais lattice site index
    Nbl=[N1,N2,N3]: Bravais lattice dimension
    bc: Boundary condition
    '''
    dictt={
    0:nr,                                          # None
    1:np.array([nr[i]%Nbl[i] for i in range(3)])    # PBC
    }
    return dictt[bc]


def pairs(r,Nbl,bc,ltype):
    '''
    The pairs between a lattice site r = [nr,sl] and itself, nearest neighbors, and second neighbors
    r=[nr,sl]: Lattice site index
    nr: Bravais lattice site index
    Nbl=[N1,N2,N3]: Bravais lattice dimension
    bc: Boundary condition
    '''
    if(ltype==221):return square.pairs(r,Nbl,bc)
    elif(ltype==233):return kagome.pairs(r,Nbl,bc)
    elif(ltype==334):return pyrochlore.pairs(r,Nbl,bc)




