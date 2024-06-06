## Lattice module

'''Lattice module: For the setup of lattices'''

from math import *
import numpy as np
import time
import joblib

import sshchain
import triangularchain
import diamondchain
import kitechain
import square
import lieb
import checkerboard
import triangular
import honeycomb
import kagome
import diamond
import pyrochlore
import tit




'''Lattice sites'''


def ltcname(ltype):
    '''
    Lattice name
    '''
    if(ltype=='sshch'):return sshchain.ltcname()
    elif(ltype=='trch'):return triangularchain.ltcname()
    elif(ltype=='dich'):return diamondchain.ltcname()
    elif(ltype=='kich'):return kitechain.ltcname()
    elif(ltype=='sq'):return square.ltcname()
    elif(ltype=='ch'):return checkerboard.ltcname()
    elif(ltype=='li'):return lieb.ltcname()
    elif(ltype=='tr'):return triangular.ltcname()
    elif(ltype=='ho'):return honeycomb.ltcname()
    elif(ltype=='ka'):return kagome.ltcname()
    elif(ltype=='dia'):return diamond.ltcname()
    elif(ltype=='py'):return pyrochlore.ltcname()
    elif(ltype=='tit'):return tit.ltcname()


def blvecs(ltype):
    '''
    Bravais lattice vectors
    '''
    if(ltype=='sshch'):return sshchain.blvecs()
    elif(ltype=='trch'):return triangularchain.blvecs()
    elif(ltype=='dich'):return diamondchain.blvecs()
    elif(ltype=='kich'):return kitechain.blvecs()
    elif(ltype=='sq'):return square.blvecs()
    elif(ltype=='ch'):return checkerboard.blvecs()
    elif(ltype=='li'):return lieb.blvecs()
    elif(ltype=='tr'):return triangular.blvecs()
    elif(ltype=='ho'):return honeycomb.blvecs()
    elif(ltype=='ka'):return kagome.blvecs()
    elif(ltype=='dia'):return diamond.blvecs()
    elif(ltype=='py'):return pyrochlore.blvecs()
    elif(ltype=='tit'):return tit.blvecs()


def slvecs(ltype):
    '''
    Sublattice vectors
    '''
    if(ltype=='sshch'):return sshchain.slvecs()
    elif(ltype=='trch'):return triangularchain.slvecs()
    elif(ltype=='dich'):return diamondchain.slvecs()
    elif(ltype=='kich'):return kitechain.slvecs()
    elif(ltype=='sq'):return square.slvecs()
    elif(ltype=='ch'):return checkerboard.slvecs()
    elif(ltype=='li'):return lieb.slvecs()
    elif(ltype=='tr'):return triangular.slvecs()
    elif(ltype=='ho'):return honeycomb.slvecs()
    elif(ltype=='ka'):return kagome.slvecs()
    elif(ltype=='dia'):return diamond.slvecs()
    elif(ltype=='py'):return pyrochlore.slvecs()
    elif(ltype=='tit'):return tit.slvecs()


def slnum(ltype):
    '''
    Sublattice number
    '''
    return np.shape(slvecs(ltype))[0]


def ltcsites(ltype,Nbl,toprint=True):
    '''
    List all of the lattice sites
    '''
    rs=[[np.array([n0,n1,n2]),sl] for n0 in range(Nbl[0]) for n1 in range(Nbl[1]) for n2 in range(Nbl[2]) for sl in range(slnum(ltype))]
    Nr=len(rs)
    if(toprint):print(ltcname(ltype),'\nSystem size =',Nbl,', site number =',Nr)
    return rs,Nr


def siteid(r,rs):
    '''
    Indices for the lattice site r in the list rs
    r=[nr,sl]: Lattice site at Bravais lattice site nr and sublattice sl
    '''
    for rn in range(len(rs)):
        rt=rs[rn]
        if(np.array_equal(r[0],rt[0]) and r[1]==rt[1]):return rn


def pos(r,ltype):
    '''
    Site position
    r=[nr,sl]: Lattice site index
    nr: Bravais lattice site index
    sl: Sublattice site index
    '''
    return np.dot(r[0],blvecs(ltype))+slvecs(ltype)[r[1]]


def cyc(nr,Nbl,bc):
    '''
    Cyclic site index for periodic boundary condition (PBC)
    nr: Bravais lattice site index
    Nbl=[N1,N2,N3]: Bravais lattice dimension
    bc: Boundary condition
    '''
    dictt={
    0:nr,   # None
    1:np.array([nr[i]%Nbl[i] for i in range(3)]),   # PBC
    111:np.array([0,0,0]),   # 1 x 1 x 1
    211:np.array([nr[0]%2,0,0]), # 2 x 1 x 1
    121:np.array([0,nr[1]%2,0]), # 2 x 1 x 1
    221:np.array([nr[0]%2,nr[1]%2,0]),   # 2 x 2 x 1
    22221:np.array({0:[0,0,0],1:[1,0,0]}[(nr[0]-nr[1])%2]),   # sqrt2 x sqrt2 x 1
    23231:np.array({0:[0,0,0],1:[1,0,0],2:[0,1,0]}[(nr[0]-nr[1])%3])   # sqrt3 x sqrt3 x 1
    }
    return dictt[bc]


'''Lattice-site connections'''


def trsl(r,ntr,Nbl,bc):
    '''
    The translation of a site r with displacement ntr on the Bravais lattice.
    '''
    return [cyc(r[0]+ntr,Nbl,bc),r[1]]


def periodictrsl(Nbl,bc,trns=[np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1])]):
    '''
    List all of the zeroth and first periodic translations under the boundary condition bc.
    '''
    nptrs=[np.array([0,0,0])]
    if(bc==1):
        if(Nbl[0]>1):nptrs+=[nptr+sgn*Nbl[0]*trns[0] for sgn in [-1,1] for nptr in nptrs]
        if(Nbl[1]>1):nptrs+=[nptr+sgn*Nbl[1]*trns[1] for sgn in [-1,1] for nptr in nptrs]
        if(Nbl[2]>1):nptrs+=[nptr+sgn*Nbl[2]*trns[2] for sgn in [-1,1] for nptr in nptrs]

    return nptrs


def pairdist(ltype,r0,r1,totrsl=False,nptrs=[]):
    '''
    Measure the distance between a pair of sites r0 and r1.
    Return: [Minimal distance, all r1s with minimal distance,r0 - minimal-distance r1]
    '''
    # No translation.
    if(totrsl==False):return np.linalg.norm(pos(r0,ltype)-pos(r1,ltype))
    # With periodic translations.
    elif(totrsl):
        # List all of the first periodic translations of r1 and compute their distances with r0.
        r1ptrs=[[r1[0]+nptr,r1[1]] for nptr in nptrs]
        rds=np.array([pairdist(ltype,r0,r1ptr) for r1ptr in r1ptrs])
        # Define the distance as the minimal result.
        rdmin=min(rds)
        r1dmids=np.argwhere(abs(rds-rdmin)<1e-12).flatten().tolist()
        r1dms=[r1ptrs[r1dmid] for r1dmid in r1dmids]
        rdvdms=[pos(r0,ltype)-pos(r1dm,ltype) for r1dm in r1dms]
        return [rdmin,r1dms,rdvdms]


def ltcpairdist(ltype,rs,Nbl,bc,toread=False,filet=''):
    '''
    List all of the pair distances on the lattice.
    Return: The matrices of neighbor indices NB, distances RD and .
    '''
    # Read from the file filet.
    if(toread==True):
        print('Read the neighbors from:',filet)
        [bc,NB,RD,RDV]=joblib.load(filet)
    else:
        # Compute all of the pair distances on the lattice.
        nptrs=periodictrsl(Nbl,bc)
        RDS=[[pairdist(ltype,r0,r1,True,nptrs) for r1 in rs] for r0 in rs]
        RD=[[RDS[rid0][rid1][0] for rid1 in range(len(rs))] for rid0 in range(len(rs))]
        RDV=[[RDS[rid0][rid1][2][0] for rid1 in range(len(rs))] for rid0 in range(len(rs))]
        # Determine the distances at the n-th neighbors.
        rds=np.sort(np.array(RD).flatten()).tolist()
        rnbs=[]
        rdt=-1.
        for rd in rds:
            if(rd-rdt>1e-12):
                rnbs+=[rd]
                rdt=rd
        rnbs=np.array(rnbs)
        # Determine the neighbor index n for all of the pairs on the lattice.
        NB=np.array([[abs(rnbs-rd).argmin() for rd in row] for row in RD])
        RD=np.array(RD)
        RDV=np.array(RDV)
    print('Boundary condition =',bc)
    return NB,RD,RDV


def nthneighbors(nb,NB):
    return np.argwhere(NB==nb).tolist()




