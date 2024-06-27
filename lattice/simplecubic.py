## Simple cubic lattice module

'''Simple cubic lattice module: Lattice structure.'''

from math import *
import numpy as np

import lattice as ltc




'''Lattice structure'''


def ltcname():
    '''
    Lattice name
    '''
    return 'Simple cubic lattice'


def blvecs():
    '''
    Bravais lattice vectors
    '''
    return np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])


def slvecs():
    '''
    Sublattice vectors
    '''
    return np.array([[0.,0.,0.]])




