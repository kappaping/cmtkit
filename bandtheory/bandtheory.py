## Band module

'''Band theory module: Setup of Hamiltonian in band theory'''

from math import *
import cmath
import numpy as np
import sympy
import joblib

import sys
sys.path.append('../lattice')
import lattice as ltc
import brillouinzone as bz
sys.path.append('../tightbinding')
import tightbinding as tb
import bogoliubovdegennes as bdg




'''Functions for Fourier transform'''


def ucsites(ltype,prds):
    '''
    List all of the lattice sites in a unit cell
    '''
    # Initialize unit-cell periodicity
    Nuc=[0,0,0]
    # Assign the periodicities for the cases with roated unit vectors. aps: Axes with rotated periodicities. p: Periodicity.
    def setperiod(aps,p):
        # sqrt2 x sqrt2
        if(p==22):
            if(len(aps)==2):dic={1:1,2:2}
        # sqrt3 x sqrt3
        elif(p==23):
            if(len(aps)==2):dic={1:1,2:3}
        # 2sqrt3 x 2sqrt3
        elif(p==223):
            if(len(aps)==2):dic={1:2,2:6}
        return [dic[(aps[n]-aps[(n+1)%len(aps)])%3] for n in range(len(aps))]
    aps=[]
    p=0
    # Assign periodicity.
    for n in range(3):
        # Set periodicity along unroated unit vectors.
        if(prds[n]<20):Nuc[n]=prds[n]
        # Find the periodicities for the rotated ones.
        elif(prds[n]>=20):
            aps+=[n]
            p=prds[n]
    ps=setperiod(aps,p)
    for n in range(len(aps)):Nuc[aps[n]]=ps[n]
    # Get a list all of the sites in the unit cell.
    rucs=[[n0*np.array([1,0,0])+n1*np.array([0,1,0])+n2*np.array([0,0,1]),sl] for n0 in range(Nuc[0]) for n1 in range(Nuc[1]) for n2 in range(Nuc[2]) for sl in range(ltc.slnum(ltype))]
    # Compute the number of lattice sites in the unit cell.
    Nruc=len(rucs)
    return rucs,Nuc,Nruc


def ucsiteid(r,prds,rucs,Nuc):
    '''
    Given the periodicity prds of a unit cell, determine the id of the site r in this unit cell.
    '''
    # If the unit cell is unrotated: Assign the ids by modulo the unit cell periodicities in each direction.
    if(max(prds)<20):return ltc.siteid([np.array([r[0][n]%Nuc[n] for n in range(3)]),r[1]],rucs)
    # sqrt2 x sqrt2
    elif(max(prds)==22):return ltc.siteid([np.array([(r[0][0]-r[0][1]%2)%2,0,0]),r[1]],rucs)
    # sqrt3 x sqrt3
    elif(max(prds)==23):return ltc.siteid([np.array([(r[0][0]-r[0][1]%3)%3,0,0]),r[1]],rucs)
    # 2sqrt3 x 2sqrt3
    elif(max(prds)==223):return ltc.siteid([np.array([(r[0][0]-2*((r[0][1]//2)%3))%6,r[0][1]%2,0]),r[1]],rucs)


def ftsites(ltype,rs,prds):
    '''
    Given a periodicity prds, determine the lattice-site pairs involved in the Fourier transform between the unit-cell-site pairs.
    Return: Nruc x Nruc list of lattice-site pairs.
    '''
    print('periodicity = ',prds)
    rucs,Nuc,Nruc=ucsites(ltype,prds)
    # List the unit-cell ids for all r in rs
    rucids=np.array([ucsiteid(r,prds,rucs,Nuc) for r in rs])
    # List the lattice-site ids for all ruc in rucs
    rucrids=[ltc.siteid(rucs[rucid],rs) for rucid in range(Nruc)]
    # Determine the list [rid0,rid1] for each site rucid0 in the unit cell 
    RUCRP=[[[[rucrids[rucid0],rid1[0]] for rid1 in np.argwhere(rucids==rucid1)] for rucid1 in range(Nruc)] for rucid0 in range(Nruc)]
    return rucs,RUCRP


def ftham(k,H,Nrfl,RDV,rucs,RUCRP,tobdg=False):
    '''
    Fourier transform of the Hamiltonian H to momentum k with a given periodicity prds.
    '''
    Nruc=len(rucs)
    if(tobdg==False):HFT=np.array([[sum([H[tb.stateid(ridp[0],fl0,Nrfl[1]),tb.stateid(ridp[1],fl1,Nrfl[1])]*e**(-1.j*np.dot(k,RDV[ridp[0],ridp[1]])) for ridp in RUCRP[rucid0][rucid1]]) for rucid1 in range(Nruc) for fl1 in range(Nrfl[1])] for rucid0 in range(Nruc) for fl0 in range(Nrfl[1])])
    elif(tobdg):
        Nst=tb.statenum(Nrfl)
        HFT=np.block([[np.array([[sum([bdg.bdgblock(H,phid0,phid1)[tb.stateid(ridp[0],fl0,Nrfl[1]),tb.stateid(ridp[1],fl1,Nrfl[1])]*e**(-1.j*np.dot(k,RDV[ridp[0],ridp[1]])) for ridp in RUCRP[rucid0][rucid1]]) for rucid1 in range(Nruc) for fl1 in range(Nrfl[1])] for rucid0 in range(Nruc) for fl0 in range(Nrfl[1])]) for phid1 in range(2)] for phid0 in range(2)])
    return HFT




'''Band properties'''


def fillingchempot(H,nf,ltype,prds,Nk):
    '''
    Compute the chemical potential for a given filling
    '''
    ks=bz.listbz(ltype,prds,Nk,True)[0]
    Hks=np.array([H(k) for k in ks])
    ees=list(np.linalg.eigvalsh(Hks).flatten())
    ees.sort()
    Nock=round(nf*len(ees))
    mu=(ees[Nock-1]+ees[Nock])/2.
    dee=ees[Nock]-ees[Nock-1]
    print('mu = ',mu,', energy gap =',dee)
    return mu


def berrycurv(k,H,dks,nf=0.,tobdg=False):
    '''
    Compute the Berry curvatures at the momentum k.
    '''
    # List the corners of the small grid around k.
    # Rectangular Brillouin zone: Cut a rectangular grid.
    if(len(dks)==2):kcts=[k+dks[0],k+dks[1],k-dks[0],k-dks[1]]
    # Hexagonal Brillouin zone: Cut a hexagonal grid.
    elif(len(dks)==3):kcts=[k+dks[0],k-dks[2],k+dks[1],k-dks[0],k+dks[2],k-dks[1]]
    # Compute the volume of the small grid.
    dka=len(dks)*np.linalg.norm(np.cross(dks[0],dks[1]))
    Nkcts=len(kcts)
    # Obtain the eigenstates at the corner momenta.
    Hkcts=[H(kct) for kct in kcts]
    Us=[np.linalg.eigh(Hkct)[1] for Hkct in Hkcts]
    # Take the inner products between adjacent corners <u(k+dk)|u(k)>.
    UTs=[U.conj().T for U in Us]
    # Extract the Berry phase of the occupied bands.
    Nbd=np.shape(Us[0])[0]
    nft=nf
    if(tobdg):nft=1./2.
    Noc=round(Nbd*nft)
    dUs=[np.dot(UTs[(nkct+1)%Nkcts][:Noc,:],Us[nkct][:,:Noc]) for nkct in range(Nkcts)]
    dUcyc=np.linalg.multi_dot(dUs)
    dBf=cmath.phase(np.linalg.det(dUcyc))
    Bf=dBf/dka
    return [dBf,Bf,dka]




