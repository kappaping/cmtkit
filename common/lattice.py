## Lattice module

'''Lattice module: For the setup of lattices'''

from math import *
import numpy as np




'''Lattice structure'''


def ltcname(mtype):
    '''
    Lattice name
    '''
    dictt={
            21:'Square lattice',     # Square
            22:'Kagome lattice',     # Kagome
            31:'Pyrochlore lattice'  # Pyrochlore
            }
    return dictt[mtype]


def avs(mtype):
    '''
    Bravais lattice vectors
    '''
    dictt={
            21:np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]), # Square
            22:np.array([[0.,2.,0.],[sqrt(3.),1.,0.],[0.,0.,1.]]),   # Kagome
            31:np.array([[0.,2.,0.],[sqrt(3.),1.,0.],[1/sqrt(3.),1.,2.*sqrt(2./3.)]])    # Pyrochlore
    }
    return dictt[mtype]


def ltcsites(Nltc):
    '''
    List all of the lattice sites
    '''
    return [[np.array([n0,n1,n2]),sl] for n0 in range(Nltc[0][0]) for n1 in range(Nltc[0][1]) for n2 in range(Nltc[0][2]) for sl in range(Nltc[1])]


def avsls(mtype):
    '''
    Sublattice vectors
    '''
    dictt={
            21:np.array([[0.,0.,0.]]),   # Square
            22:np.array([[0.,0.,0.],[0.,1.,0.],[sqrt(3.)/2.,1./2.,0.]]), # Kagome
            31:np.array([[0.,0.,0.],[0.,1.,0.],[sqrt(3.)/2.,1./2.,0.],[1./(2.*sqrt(3.)),1./2.,sqrt(2./3.)]]) # Pyrochlore
    }
    return dictt[mtype]


def nslf(mtype):
    '''
    Sublattice number
    '''
    return np.shape(avsls(mtype))[0]


def rid(r,Nltc):
    '''
    Indices for the lattice site r
    r=[nr,sl]: Lattice site at Bravais lattice site nr and sublattice sl
    Nltc=[Nbl,Nsl]: Bravais lattice dimension, sublattice number
    '''
    return Nltc[1]*(Nltc[0][2]*(Nltc[0][1]*r[0][0]+r[0][1])+r[0][2])+r[1]


def pos(r,mtype):
    '''
    Site position
    r=[nr,sl]: Lattice site index
    nr: Bravais lattice site index
    sls: Sublattice site index
    '''
    return np.dot(np.array(r[0]),avs(mtype))+avsls(mtype)[r[1]]


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


def pairs(r,Nbl,bc,mtype):
    '''
    The pairs between a lattice site r = [nr,sl] and itself, nearest neighbors, and second neighbors
    r=[nr,sl]: Lattice site index
    nr: Bravais lattice site index
    Nbl=[N1,N2,N3]: Bravais lattice dimension
    bc: Boundary condition
    '''
    nr=r[0]
    sl=r[1]

    # ONblite
    pairs0th=[[[nr,sl],[nr,sl]]]

    # Nearest neighbors
    pairs1st={
        # Square
        21:{
            0:[
                [[nr,0],[cyc(nr+np.array([-1,0,0]),Nbl,bc),0]],[[nr,0],[cyc(nr+np.array([1,0,0]),Nbl,bc),0]],
                [[nr,0],[cyc(nr+np.array([0,-1,0]),Nbl,bc),0]],[[nr,0],[cyc(nr+np.array([0,1,0]),Nbl,bc),0]]
                ]
            },
       # Kagome
        22:{
            # sl1=0
            0:[[[nr,0],[nr,1]],[[nr,0],[nr,2]],[[nr,0],[cyc(nr+np.array([-1,0,0]),Nbl,bc),1]],[[nr,0],[cyc(nr+np.array([0,-1,0]),Nbl,bc),2]]],
            # sl1=1
            1:[[[nr,1],[nr,0]],[[nr,1],[nr,2]],[[nr,1],[cyc(nr+np.array([1,0,0]),Nbl,bc),0]],[[nr,1],[cyc(nr+np.array([1,-1,0]),Nbl,bc),2]]],
            # sl1=2
            2:[[[nr,2],[nr,0]],[[nr,2],[nr,1]],[[nr,2],[cyc(nr+np.array([0,1,0]),Nbl,bc),0]],[[nr,2],[cyc(nr+np.array([-1,1,0]),Nbl,bc),1]]]
            },
        # Pyrochlore
        31:{
            # sl1=0
            0:[
                [[nr,0],[nr,1]],[[nr,0],[nr,2]],[[nr,0],[nr,3]],
                [[nr,0],[cyc(nr+np.array([-1,0,0]),Nbl,bc),1]],
                [[nr,0],[cyc(nr+np.array([0,-1,0]),Nbl,bc),2]],
                [[nr,0],[cyc(nr+np.array([0,0,-1]),Nbl,bc),3]]
                ],
            # sl1=1
            1:[
                [[nr,1],[nr,0]],[[nr,1],[nr,2]],[[nr,1],[nr,3]],
                [[nr,1],[cyc(nr+np.array([1,0,0]),Nbl,bc),0]],
                [[nr,1],[cyc(nr+np.array([1,-1,0]),Nbl,bc),2]],
                [[nr,1],[cyc(nr+np.array([1,0,-1]),Nbl,bc),3]]
                ],
            # sl1=2
            2:[
                [[nr,2],[nr,0]],[[nr,2],[nr,1]],[[nr,2],[nr,3]],
                [[nr,2],[cyc(nr+np.array([0,1,0]),Nbl,bc),0]],
                [[nr,2],[cyc(nr+np.array([-1,1,0]),Nbl,bc),1]],
                [[nr,2],[cyc(nr+np.array([0,1,-1]),Nbl,bc),3]]
                ],
            # sl1=3
            3:[
                [[nr,3],[nr,0]],[[nr,3],[nr,1]],[[nr,3],[nr,2]],
                [[nr,3],[cyc(nr+np.array([0,0,1]),Nbl,bc),0]],
                [[nr,3],[cyc(nr+np.array([-1,0,1]),Nbl,bc),1]],
                [[nr,3],[cyc(nr+np.array([0,-1,1]),Nbl,bc),2]]
                ]
            }
        }

    # Second neighbors
    pairs2nd={
        # Square
        21:{
            0:[
                [[nr,0],[cyc(nr+np.array([-1,-1,0]),Nbl,bc),0]],[[nr,0],[cyc(nr+np.array([-1,1,0]),Nbl,bc),0]],
                [[nr,0],[cyc(nr+np.array([1,-1,0]),Nbl,bc),0]],[[nr,0],[cyc(nr+np.array([1,1,0]),Nbl,bc),0]]
                ]
            },
        # Kagome
        22:{
            # sl1=0
            0:[
                [[nr,0],[cyc(nr+np.array([0,-1,0]),Nbl,bc),1]],[[nr,0],[cyc(nr+np.array([1,-1,0]),Nbl,bc),2]],
                [[nr,0],[cyc(nr+np.array([-1,1,0]),Nbl,bc),1]],[[nr,0],[cyc(nr+np.array([-1,0,0]),Nbl,bc),2]]
                ],
            # sl1=1
            1:[
                [[nr,1],[cyc(nr+np.array([1,-1,0]),Nbl,bc),0]],[[nr,1],[cyc(nr+np.array([0,-1,0]),Nbl,bc),2]],
                [[nr,1],[cyc(nr+np.array([0,1,0]),Nbl,bc),0]],[[nr,1],[cyc(nr+np.array([1,0,0]),Nbl,bc),2]]
                ],
            # sl1=2
            2:[
                [[nr,2],[cyc(nr+np.array([1,0,0]),Nbl,bc),0]],[[nr,2],[cyc(nr+np.array([0,1,0]),Nbl,bc),1]],
                [[nr,2],[cyc(nr+np.array([-1,1,0]),Nbl,bc),0]],[[nr,2],[cyc(nr+np.array([-1,0,0]),Nbl,bc),1]]
                ]
            },
        # Pyrochlore
        31:{}
        }

    return [pairs0th,pairs1st[mtype][sl],pairs2nd[mtype][sl]]




