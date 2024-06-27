## TaIrTe4 lattice module

'''TaIrTe4 lattice module: Lattice structure.'''

from math import *
import numpy as np

import lattice as ltc




'''Lattice structure'''


def ltcname():
    '''
    Lattice name
    '''
    return 'TaIrTe4 lattice'


def blvecs():
    '''
    Bravais lattice vectors
    '''
    return np.array([[2.*cos(3.*pi/8.),0.,0.],[0.,3.,0.],[0.,0.,1.]])


def slvecs():
    '''
    Sublattice vectors
    '''
    return np.array([[0.,0.,0.],[cos(3.*pi/8.),sin(3.*pi/8.),0.]])




