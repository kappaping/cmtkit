## Band module

'''Band theory module: Setup of Hamiltonian in band theory'''

from math import *
import cmath as cmt
import numpy as np
import sympy

import sys
sys.path.append('../lattice')
import lattice as ltc
sys.path.append('../tightbinding')
import tightbinding as tb




'''Matrix setup'''


def ucsites(ltype,uctype):
    '''
    List all of the lattice sites in a unit cell
    '''
    dictt={
    111:[np.array([0,0,0])],   # 1 x 1 x 1
    211:[np.array([n0,0,0]) for n0 in range(2)], # 2 x 1 x 1
    121:[np.array([0,n1,0]) for n1 in range(2)], # 2 x 1 x 1
    221:[np.array([n0,n1,0]) for n0 in range(2) for n1 in range(2)],   # 2 x 2 x 1
    22221:[np.array([n0,0,0]) for n0 in range(2)],  # sqrt2 x sqrt2 x 1
    23231:[np.array([n0,n1,0]) for n0 in range(2) for n1 in range(2-n0)]  # sqrt3 x sqrt3 x 1
    }
    return [[nr,sl] for nr in dictt[uctype] for sl in range(ltc.slnum(ltype))]


def ucnums(uctype):
    '''
    Give the dimensions of the Bravais lattice unit cells
    '''
    dictt={
    111:[1,1,1],  # 1 x 1 x 1
    211:[2,1,1], # 2 x 1 x 1
    121:[1,2,1], # 2 x 1 x 1
    221:[2,2,1], # 2 x 2 x 1
    22221:[2,1,1], # sqrt2 x sqrt2 x 1
    23231:[2,2,1] # sqrt3 x sqrt3 x 1
    }
    return dictt[uctype]


def ucstnum(ltype,uctype,Nfl):
    '''
    State number in the unit cell
    '''
    return len(ucsites(ltype,uctype))*Nfl


def tbham(k,htb,ltype,uctype,Nfl):
    '''
    Tight-binding model in momentum space: Assign the couplings htb=[v0,-t1,-t2] to the Hamiltonian H
    v0: Onsite potential
    t1 and t2: Nearest and second neighbor hoppings
    The factor 1/2 is to cancel double counting from the Hermitian assignment in termmat
    '''
    Nalluc=[[ucnums(uctype),ltc.slnum(ltype)],Nfl]
    rsuc=ucsites(ltype,uctype)
    H=np.zeros((ucstnum(ltype,uctype,Nfl),ucstnum(ltype,uctype,Nfl)),dtype=complex)
    for r in rsuc:
        # Pairs at Bravais lattice site r
        pairs0=ltc.pairs(r,ucnums(uctype),0,ltype)
        # Pairs in the unit cell
        pairsuc=ltc.pairs(r,ucnums(uctype),uctype,ltype)
        # Add matrix elements for the pairs
        [tb.termmat(H,(1./2.)*htb[nd]*e**(-1.j*np.dot(k,(ltc.pos(pairs0[nd][npr][0],ltype)-ltc.pos(pairs0[nd][npr][1],ltype)))),pairsuc[nd][npr][0],fl,pairsuc[nd][npr][1],fl,Nalluc) for nd in range(len(pairs0)) for npr in range(len(pairs0[nd])) for fl in range(Nfl)]
    return H




'''Momentum space setup'''


def hskcontour(ltype,uctype):
    '''
    Set the high-symmetry points in the Brillouin zone forming the contour for the band structure.
    '''
    hska=ltc.hskpoints(ltype,uctype)
    if(ltype=='sq' and (uctype==111 or uctype==221)):
        return [hska[0],hska[1],hska[3],hska[2],hska[0]]
    if(uctype==211 or uctype==121):
        return [hska[0],hska[1],hska[3],hska[0],hska[2],hska[3],hska[0]]
    elif((ltype=='tr' or ltype=='ka') and (uctype==111 or uctype==221)):
        return [hska[0],hska[1],[hska[5][0],-hska[5][1]],hska[0]]
    elif((ltype=='tr' or ltype=='ka') and uctype==23231):
        return [hska[0],hska[1],[hska[5][0],hska[5][1]],hska[0]]


def brillouinzone(ltype,uctype,Nk,bzop=False):
    '''
    The momenta in the Brillouin zone.
    '''
    ks=[]
    dks=[]
    if(bzop==True): dkb=1e-12
    else: dkb=0.
    if(ltype=='sq' or uctype==211 or uctype==121):
        kcs=[ltc.hskpoints(ltype,uctype)[n][1] for n in [1,2]]
        g0,g1=kcs[0],kcs[1]
        for n0 in np.linspace(-2.,2.,num=2*Nk+1):
            for n1 in np.linspace(-2.,2.,num=2*Nk+1):
                k=n0*g0+n1*g1
                if(-np.linalg.norm(kcs[0])**2-1e-14<=np.dot(k,kcs[0])<np.linalg.norm(kcs[0])**2+1e-14-dkb and -np.linalg.norm(kcs[1])**2-1e-14<=np.dot(k,kcs[1])<np.linalg.norm(kcs[1])**2+1e-14-dkb):
                    ks.append(k)
        dks=[(1./(2.*Nk))*ltc.hskpoints(ltype,uctype)[n][1] for n in [3,4]]
    elif((ltype=='tr' or ltype=='ka') and (uctype==111 or uctype==221 or uctype==23231)):
        kcs=[ltc.hskpoints(ltype,uctype)[n][1] for n in [1,2,3]]
        g0,g1=kcs[0],kcs[1]
        for n0 in np.linspace(-2.,2.,num=2*Nk+1):
            for n1 in np.linspace(-2.,2.,num=2*Nk+1):
                k=n0*g0+n1*g1
                if(-np.linalg.norm(kcs[0])**2-1e-14<=np.dot(k,kcs[0])<np.linalg.norm(kcs[0])**2+1e-14-dkb and -np.linalg.norm(kcs[1])**2-1e-14<=np.dot(k,kcs[1])<np.linalg.norm(kcs[1])**2+1e-14-dkb and -np.linalg.norm(kcs[2])**2-1e-14<=np.dot(k,kcs[2])<np.linalg.norm(kcs[2])**2+1e-14-dkb):
                    ks.append(k)
        dks=[(1./(2.*Nk))*ltc.hskpoints(ltype,uctype)[n][1] for n in [4,5,6]]
    return [ks,dks]








