## Diamond lattice module

'''Diamond lattice module: Lattice structure.'''

from math import *
import numpy as np

import lattice as ltc




'''Lattice structure'''


def ltcname():
    '''
    Lattice name
    '''
    return 'Diamond lattice'


def blvecs():
    '''
    Bravais lattice vectors
    '''
    return np.array([[0.,2.*sqrt(2./3.),0.],[sqrt(2.),sqrt(2./3.),0.],[sqrt(2.)/3.,sqrt(2./3.),4./3.]])


def slvecs():
    '''
    Sublattice vectors
    '''
    return np.array([[0.,0.,0.],sum(list(blvecs()))/4.])




