## Lieb lattice module

'''Lieb lattice module: Lattice structure.'''

from math import *
import numpy as np

import lattice as ltc




'''Lattice structure'''


def ltcname():
    '''
    Lattice name
    '''
    return 'Lieb lattice'


def blvecs():
    '''
    Bravais lattice vectors
    '''
    return np.array([[2.,0.,0.],[0.,2.,0.],[0.,0.,1.]])


def slvecs():
    '''
    Sublattice vectors
    '''
    return np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])




