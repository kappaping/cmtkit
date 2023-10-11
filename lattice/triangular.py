## Triangular lattice module

'''Triangular lattice module: Structure of triangular lattice'''

from math import *
import numpy as np

import lattice as ltc




'''Lattice structure'''


def ltcname():
    '''
    Lattice name
    '''
    return 'Triangular lattice'


def blvecs():
    '''
    Bravais lattice vectors
    '''
    return np.array([[0.,1.,0.],[sqrt(3.)/2.,1./2.,0.],[0.,0.,1.]])


def slvecs():
    '''
    Sublattice vectors
    '''
    return np.array([[0.,0.,0.]])




