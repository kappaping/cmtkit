## Hartree-Fock module

'''Hartree-Fock module: Functions of Hartree-Fock approximation'''

from math import *
import cmath as cmt
import numpy as np

import sys
sys.path.append('../lattice')
import lattice as ltc




'''Matrix setup'''




def vhartreefock(Mt,Pt,uhs,rs,Nall,bc,ltype):
    '''
    Hartree-Fock potential: Assign the Hartree-Fock potentials from the interaction htb=[u0,u1,u2] to the Hamiltonian Mt.
    The factor 1/2 is to cancel double counting from the Hermitian assignment in termmat
    '''
    for r in rs:
        # Pairs at Bravais lattice site bls
        pairst=ltc.pairs(r,Nall[0][0],bc,ltype)
        # Add Hartree potential
        [termmat(Mt,(1./2.)*uhs[nd]*Pt[stid(pairt[1],fl1,Nall),stid(pairt[1],fl1,Nall)],pairt[0],fl0,pairt[0],fl0,Nall) for nd in range(len(pairst)) for pairt in pairst[nd] for fl0 in range(Nall[1]) for fl1 in range(Nall[1])]
        # Add Fock potential
        [termmat(Mt,(-1./2.)*uhs[nd]*Pt[stid(pairt[0],fl0,Nall),stid(pairt[1],fl1,Nall)],pairt[0],fl0,pairt[1],fl1,Nall) for nd in range(len(pairst)) for pairt in pairst[nd] for fl0 in range(Nall[1]) for fl1 in range(Nall[1])]




def tbham(Mt,htb,rs,Nall,bc,ltype):
    '''
    Tight-binding model: Assign the couplings htb=[v0,-t1,-t2] to the Hamiltonian Mt.
    v0: Onsite potential
    t1 and t2: Nearest and second neighbor hoppings
    The factor 1/2 is to cancel double counting from the Hermitian assignment in termmat
    '''
    for r in rs:
        # Pairs at Bravais lattice site bls
        pairst=ltc.pairs(r,Nall[0][0],bc,ltype)
        # Add matrix elements for the pairs
        [termmat(Mt,(1./2.)*htb[nd],pairt[0],fl,pairt[1],fl,Nall) for nd in range(len(pairst)) for pairt in pairst[nd] for fl in range(Nall[1])]



