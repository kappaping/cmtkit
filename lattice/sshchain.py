## Su-Schrieffer-Heeger chain lattice module

'''Su-Schrieffer-Heeger chain lattice module: Structure of Su-Schrieffer-Heeger lattice'''

from math import *
import numpy as np

import lattice as ltc




'''Lattice structure'''


def ltcname():
    '''
    Lattice name
    '''
    return 'Su-Schrieffer-Heeger chain lattice'


def blvecs():
    '''
    Bravais lattice vectors
    '''
    return np.array([[sqrt(3.),0.,0.],[0.,1.,0.],[0.,0.,1.]])


def slvecs():
    '''
    Sublattice vectors
    '''
    return np.array([[0.,0.,0.],[sqrt(3.)/2.,1./2.,0.]])




