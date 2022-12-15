## Lattice module

'''Lattice module: For the setup of lattices'''

from math import *
import numpy as np

import square
import triangular
import kagome
import pyrochlore




'''Lattice structure'''


def ltcname(ltype):
    '''
    Lattice name
    '''
    if(ltype=='sq'):return square.ltcname()
    elif(ltype=='tr'):return triangular.ltcname()
    elif(ltype=='ka'):return kagome.ltcname()
    elif(ltype=='py'):return pyrochlore.ltcname()


def blvecs(ltype):
    '''
    Bravais lattice vectors
    '''
    if(ltype=='sq'):return square.blvecs()
    elif(ltype=='tr'):return triangular.blvecs()
    elif(ltype=='ka'):return kagome.blvecs()
    elif(ltype=='py'):return pyrochlore.blvecs()


def ltcsites(Nltc):
    '''
    List all of the lattice sites
    '''
    return [[np.array([n0,n1,n2]),sl] for n0 in range(Nltc[0][0]) for n1 in range(Nltc[0][1]) for n2 in range(Nltc[0][2]) for sl in range(Nltc[1])]


def slvecs(ltype):
    '''
    Sublattice vectors
    '''
    if(ltype=='sq'):return square.slvecs()
    elif(ltype=='tr'):return triangular.slvecs()
    elif(ltype=='ka'):return kagome.slvecs()
    elif(ltype=='py'):return pyrochlore.slvecs()


def slnum(ltype):
    '''
    Sublattice number
    '''
    return np.shape(slvecs(ltype))[0]


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
    return np.dot(np.array(r[0]),blvecs(ltype))+slvecs(ltype)[r[1]]


def cyc(nr,Nbl,bc):
    '''
    Cyclic site index for periodic boundary condition (PBC)
    nr: Bravais lattice site index
    Nbl=[N1,N2,N3]: Bravais lattice dimension
    bc: Boundary condition
    '''
    dictt={
    0:nr,   # None
    1:np.array([nr[i]%Nbl[i] for i in range(3)]),   # PBC
    111:np.array([0,0,0]),   # 1 x 1 x 1
    211:np.array([nr[0]%2,0,0]), # 2 x 1 x 1
    221:np.array([nr[0]%2,nr[1]%2,0]),   # 2 x 2 x 1
    331:np.array({0:[0,0,0],1:[1,0,0],2:[0,1,0]}[(nr[0]-nr[1])%3])   # sqrt3 x sqrt3 x 1
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
    if(ltype=='sq'):return square.pairs(r,Nbl,bc)
    elif(ltype=='tr'):return triangular.pairs(r,Nbl,bc)
    elif(ltype=='ka'):return kagome.pairs(r,Nbl,bc)
    elif(ltype=='py'):return pyrochlore.pairs(r,Nbl,bc)


'''Original Brillouin zone'''


def hskpoints(ltype):
    '''
    List the high-symmetry points of the Brillouin zone
    '''
    # Square lattice: [Gamma,X,Y,M1,M2]
    if(ltype=='sq'):return square.hskpoints()
    # Triangular lattice: [Gamma,M1,M2,M3,K1,K2,K3]
    elif(ltype=='tr'):return triangular.hskpoints()
    # Kagome lattice: [Gamma,M1,M2,M3,K1,K2,K3]
    elif(ltype=='ka'):return kagome.hskpoints()




