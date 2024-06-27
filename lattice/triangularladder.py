## Triangular-ladder lattice module

'''Triangular-ladder lattice module: Lattice structure.'''

from math import *
import numpy as np

import lattice as ltc




'''Lattice structure'''


def ltcname():
    '''
    Lattice name
    '''
    return 'Triangular chain lattice'


def blvecs():
    '''
    Bravais lattice vectors
    '''
    return np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])


def slvecs():
    '''
    Sublattice vectors
    '''
    return np.array([[0.,0.,0.],[1./2.,sqrt(3.)/2.,0.]])




