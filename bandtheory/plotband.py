## Plot band module

'''Plot band module: Functions of plotting the bands'''

from math import *
import cmath as cmt
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../lattice')
import lattice as ltc
sys.path.append('../tightbinding')
import tightbinding as tb
import bandtheory as bdth




'''Original Brillouin zone'''


def sectionband(H,k1,k2,k0,Nk,ks,bands):
    '''
    Compute the band eigenvalues along a momentum-space line section k1-k2.
    '''
    k12s=list(np.linspace(k1,k2,num=Nk))
    dk=np.linalg.norm(k12s[1]-k12s[0])
    for nk in range(len(k12s)):
        ks.append(k0+nk*dk)
        Hk=H(k12s[nk])
        ees=np.linalg.eigvalsh(Hk)
        for n in range(len(ees)):
            bands[n].append(ees[n])


def plotband(H,Nbd,hsks,Nk):
    '''
    Plot the band structure along a trajectory in the Brillouin zone.
    '''
    ks=[]
    bands=[[] for n in range(Nbd)]
    k0=0.
    for n in range(len(hsks)-1):
        sectionband(H,hsks[n],hsks[n+1],k0,Nk,ks,bands)
        k0+=np.linalg.norm(hsks[n+1]-hsks[n])
    for n in range(Nbd):
        plt.plot(ks,bands[n])
    plt.xlabel('k')
    #plt.legend()
    plt.show()
#    plt.savefig('test.png')










