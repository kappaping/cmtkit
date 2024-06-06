## Diamond chain lattice module

'''Diamond chain lattice module: Structure of Diamond lattice'''

from math import *
import numpy as np

import lattice as ltc




'''Lattice structure'''


def ltcname():
    '''
    Lattice name
    '''
    return 'Diamond chain lattice'


def blvecs():
    '''
    Bravais lattice vectors
    '''
    return np.array([[sqrt(2.),0.,0.],[0.,1.,0.],[0.,0.,1.]])


def slvecs():
    '''
    Sublattice vectors
    '''
    return np.array([[0.,0.,0.],[1./sqrt(2.),1./sqrt(2.),0.],[1./sqrt(2.),-1./sqrt(2.),0.]])




