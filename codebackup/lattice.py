## Lattice module

'''Lattice module: For the setup of lattices'''

from math import *
import numpy as np

import square
import triangular
import kagome
import pyrochlore




'''Lattice sites'''


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
    121:np.array([0,nr[1]%2,0]), # 2 x 1 x 1
    221:np.array([nr[0]%2,nr[1]%2,0]),   # 2 x 2 x 1
    22221:np.array({0:[0,0,0],1:[1,0,0]}[(nr[0]-nr[1])%2]),   # sqrt2 x sqrt2 x 1
    23231:np.array({0:[0,0,0],1:[1,0,0],2:[0,1,0]}[(nr[0]-nr[1])%3])   # sqrt3 x sqrt3 x 1
    }
    return dictt[bc]


'''Lattice-site connections'''


def trsl(r,ntr,Nbl,bc):
    '''
    The translation of a site r with displacement ntr on the Bravais lattice.
    '''
    return [cyc(r[0]+ntr,Nbl,bc),r[1]]


def pairdist(ltype,r0,r1,Nbl,bc):
    '''
    Measure the distance between a pair of sites r0 and r1.
    '''
    # List all of the first periodic translations.
    nptrs=[np.array([0,0,0])]
    if(bc==1):
        if(Nbl[0]>1):nptrs+=[nptr+sgn*np.array([Nbl[0],0,0]) for sgn in [-1,1] for nptr in nptrs]
        if(Nbl[1]>1):nptrs+=[nptr+sgn*np.array([0,Nbl[1],0]) for sgn in [-1,1] for nptr in nptrs]
        if(Nbl[2]>1):nptrs+=[nptr+sgn*np.array([0,0,Nbl[2]]) for sgn in [-1,1] for nptr in nptrs]
    # List all of the first periodic translations of r1 and compute their distances with r0.
    r1ptrs=[trsl(r1,nptr,Nbl,0) for nptr in nptrs]
    rds=np.array([np.linalg.norm(pos(r0,ltype)-pos(r1ptr,ltype)) for r1ptr in r1ptrs])
    # Define the distance as the minimal result.
    rd=np.amin(rds)

    return rd


def ltcpairdist(ltype,rs,Nbl,bc):
    '''
    List all of the pair distances on the lattice. Return the matrices of neighbor indices NB and distances RD.
    '''
    # Compute all of the pair distances on the lattice.
    RD=np.array([[pairdist(ltype,r0,r1,Nbl,bc) for r0 in rs] for r1 in rs])
    # Determine the distances at the n-th neighbors.
    rds=sorted(list(RD[0]))
    rnbs=[]
    rdt=-1.
    for rd in rds:
        if(rd-rdt>1e-12):
            rnbs+=[rd]
            rdt=rd
    # Determine the neighbor index n for all of the pairs on the lattice.
    def neighborid(rd,rnbs):
        for nb in range(len(rnbs)):
            if(abs(rd-rnbs[nb]<1e-12)):
                return nb
    NB=np.array([[neighborid(rd,rnbs) for rd in row] for row in RD])

    return NB,RD


''''''


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


def hskpoints(ltype,uctype):
    '''
    List the high-symmetry points of the Brillouin zone
    '''
    # Square lattice: [Gamma,X,Y,M1,M2]
    if(ltype=='sq'):return square.hskpoints(uctype)
    # Triangular lattice: [Gamma,M1,M2,M3,K1,K2,K3]
    elif(ltype=='tr'):return triangular.hskpoints(uctype)
    # Kagome lattice: [Gamma,M1,M2,M3,K1,K2,K3]
    elif(ltype=='ka'):return kagome.hskpoints(uctype)




