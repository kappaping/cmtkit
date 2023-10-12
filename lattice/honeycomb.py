## Honeycomb lattice module

'''Honeycomb lattice module: Structure of honeycomb lattice'''

from math import *
import numpy as np

import lattice as ltc




'''Lattice structure'''


def ltcname():
    '''
    Lattice name
    '''
    return 'Honeycomb lattice'


def blvecs():
    '''
    Bravais lattice vectors
    '''
    return np.array([[0.,sqrt(3.),0.],[3./2.,sqrt(3.)/2.,0.],[0.,0.,1.]])


def slvecs():
    '''
    Sublattice vectors
    '''
    return np.array([[0.,0.,0.],[1.,0.,0.]])




