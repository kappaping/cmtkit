## Square lattice module

'''Square lattice module: Structure of square lattice'''

from math import *
import numpy as np

import lattice as ltc




'''Lattice structure'''


def ltcname():
    '''
    Lattice name
    '''
    return 'Square lattice'


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




