## Body-centered cubic lattice 0 module

'''Body-centered cubic lattice 0 module: Lattice structure.'''

from math import *
import numpy as np

import lattice as ltc




'''Lattice structure'''


def ltcname():
    '''
    Lattice name
    '''
    return 'Body-centered cubic lattice (with simple cubic Bravais lattice)'


def blvecs():
    '''
    Bravais lattice vectors
    '''
    return np.array([[2./sqrt(3.),0.,0.],[0.,2./sqrt(3.),0.],[0.,0.,2./sqrt(3.)]])


def slvecs():
    '''
    Sublattice vectors
    '''
    return np.array([[0.,0.,0.],[1./sqrt(3.),1./sqrt(3.),1./sqrt(3.)]])




