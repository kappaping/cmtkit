## Face-centered cubic lattice 0 module

'''Face-centered cubic lattice 0 module: Lattice structure.'''

from math import *
import numpy as np

import lattice as ltc




'''Lattice structure'''


def ltcname():
    '''
    Lattice name
    '''
    return 'Face-centered cubic lattice (with simple cubic Bravais lattice)'


def blvecs():
    '''
    Bravais lattice vectors
    '''
    return np.array([[sqrt(2.),0.,0.],[0.,sqrt(2.),0.],[0.,0.,sqrt(2.)]])


def slvecs():
    '''
    Sublattice vectors
    '''
    return np.array([[0.,0.,0.],[0.,1./sqrt(2.),1./sqrt(2.)],[1./sqrt(2.),0.,1./sqrt(2.)],[1./sqrt(2.),1./sqrt(2.),0.]])




