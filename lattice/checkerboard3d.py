## Checkerboard lattice module

'''Checkerboard lattice module: Lattice structure.'''

from math import *
import numpy as np

import lattice as ltc




'''Lattice structure'''


def ltcname():
    '''
    Lattice name
    '''
    return 'Checkerboard lattice 3D'


def blvecs():
    '''
    Bravais lattice vectors
    '''
    return np.array([[1.,1.,0.],[1.,0.,1.],[0.,1.,1.]])


def slvecs():
    '''
    Sublattice vectors
    '''
    return np.array([[0.,0.,0.],[0.,1.,0.]])




