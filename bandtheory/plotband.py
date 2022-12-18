## Plot band module

'''Plot band module: Functions of plotting the bands'''

from math import *
import cmath as cmt
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size']=18
plt.rcParams.update({'figure.autolayout': True})

import sys
sys.path.append('../lattice')
import lattice as ltc
sys.path.append('../tightbinding')
import tightbinding as tb
import bandtheory as bdth




'''Original Brillouin zone'''


def sectionband(H,mu,k1,k2,k0,Nk,toend,ks,bands,cs):
    '''
    Compute the band eigenvalues along a momentum-space line section k1-k2.
    '''
    k12s=list(np.linspace(k1,k2,num=Nk,endpoint=toend))
    dk=np.linalg.norm(k12s[1]-k12s[0])
    for nk in range(len(k12s)):
        ks.append(k0+nk*dk)
        Hk=H(k12s[nk])
        ees=np.linalg.eigvalsh(Hk)
        for n in range(len(ees)):
            bands[n].append(ees[n])
            if(ees[n]<mu+1e-14):
                cs[n].append('g')
            else:
                cs[n].append('b')


def plotband(H,mu,Nbd,hsks,Nk):
    '''
    Plot the band structure along a trajectory in the Brillouin zone.
    '''
    ks=[]
    bands=[[] for n in range(Nbd)]
    cs=[[] for n in range(Nbd)]
    k0=0.
    kts,ktlbs=[k0],[hsks[0][0]]
    for ns in range(len(hsks)-1):
        if(ns==len(hsks)-2):toend=True
        else:toend=False
        sectionband(H,mu,hsks[ns][1],hsks[ns+1][1],k0,Nk,toend,ks,bands,cs)
        k0+=np.linalg.norm(hsks[ns+1][1]-hsks[ns][1])
        kts.append(k0)
        ktlbs.append(hsks[ns+1][0])
    for n in range(Nbd):
        plt.scatter(ks,bands[n],s=2.,c=cs[n])
    [plt.axvline(x=hsk,color='k') for hsk in kts[1:-1]]
    plt.xlim(kts[0],kts[-1])
    plt.xticks(ticks=kts,labels=ktlbs)
    plt.ylabel('Ek')
    plt.gcf()
    plt.show()
#    plt.savefig('test.png')










