## Density-matrix module

'''Density-matrix module: Manipulations of density matrix.'''

from math import *
import numpy as np
from scipy.stats import unitary_group
import joblib

import sys
sys.path.append('../lattice')
import lattice as ltc
import tightbinding as tb




'''Setup of density matrix'''


def projdenmat(U,n0,n1,Nst):
    '''
    Generate the density matrix by projecting on the n0-th to n1-th states of the unitary operator U=[u0,u1,u2,....].
    '''
    UT=U.conj().T
    # Project to only the Noc occupied states
    D=np.diag(np.array(n0*[0.]+(n1-n0)*[1.]+(Nst-n1)*[0.]))
    return np.round(np.linalg.multi_dot([U,D,UT]),25)


def setdenmat(Ptype,Nrfl,nf,fileti='',ltype='',rs=[],Nbl=[],NB=np.array([]),Nbli=[],toptb=False,ptb=0.01,toflrot=False,Ufl=np.array([[0.965926-0.12941j,-0.194114-0.112072j],[0.194114-0.112072j,0.965926+0.12941j]]),tobdg=False):
    '''
    Set up a density matrix.
    Return: A density matrix.
    Ptype: Type of setting the density matrix.
    Nrfl=[Nr,Nfl]: [site number, flavor number].
    nf: Filling fraction of each state.
    fileti: File name for reading the density matrix.
    ltype: Lattice type.
    rs: List of lattice sites.
    Nbl: Bravais-lattice dimensions.
    NB: NrxNr matrix of neighbor indices.
    Nbli: Bravais-lattice dimensions of the initial density matrix.
    toptb: To perturb the density matrix.
    toflrot: To rotate the flavor
    '''
    # Initialize the density matrix.
    Nst=tb.statenum(Nrfl)
    P=np.zeros((Nst,Nst),dtype=complex)
    if(tobdg):P=np.block([[P,P],[P,P]])
    # Compute the occupation number.
    print('Filling =',nf)
    Noc=round(Nst*nf)
    # Set the density matrix by the assigned type Ptype.
    # Randomize the density matrix with occupation number Noc.
    if(Ptype=='rand'):
        print('Get a random density matrix')
        P=projdenmat(unitary_group.rvs(Nst),0,Noc,Nst)
        if(tobdg):
            PBs=[[P,P],[P,-P.T]]
            P01=unitary_group.rvs(Nst)
            P01=(P01-P01.T)/2.
            PBs[0][1]=P01
            PBs[1][0]=P01.conj().T
            P=np.block(PBs)
    # Read the density matrix from the file fileti.
    elif(Ptype=='read'):
        print('Read the density matrix from:', fileti)
        P=joblib.load(fileti)
    # Copy the density matrix from the one in file fileti with size Nbli under periodic boundary condition.
    elif(Ptype=='copy'):
        print('Copy the density matrix with system size',Nbli,'from:', fileti)
        P=denmatcopy(ltype,rs,Nrfl,Nbl,NB,fileti,Nbli)
    # Others: The density matrix is assigned by other functions.
    else:
        print('Assign the density matrix as:', Ptype)
        # Add a uniform density distribution with filling nf to the density matrix.
        [tb.termmat(P,(1./2.)*nf,rid,fl,rid,fl,Nrfl[1]) for rid in range(Nrfl[0]) for fl in range(Nrfl[1])]
    if(toptb):
        print('Perturb the density matrix at scale =',ptb)
        P=(1.-ptb)*P+ptb*projdenmat(unitary_group.rvs(Nst),0,Noc,Nst)
    if(toflrot):
        print('Rotate the flavors of the density matrix')
        P=flrot(P,Nrfl,Ufl)

    return P


def denmatcopy(ltype,rs,Nrfl,Nbl,NB,fileti,Nbli):
    '''
    Density matrix copy: Copy the density matrix from the one in file fileti with size Nbli under periodic boundary condition.
    Return: A density matrix with size Nrfl.
    ltype: Lattice type.
    rs: List of lattice sites.
    Nrfl=[Nr,Nfl]: [site number, flavor number].
    Nbl: Bravais-lattice dimensions.
    NB: NrxNr matrix of neighbor indices.
    fileti: File name of read initial density matrix.
    Nbli: Bravais-lattice dimensions of the initial density matrix.
    '''
    # Lattice sites for the initial density matrix.
    ris=ltc.ltcsites(ltype,Nbli,toprint=False)[0]
    # Find out the ids of lattice sites rs in the initial lattice ris.
    rcpids=[ltc.siteid([[r[0][n]%Nbli[n] for n in range(3)],r[1]],ris) for r in rs]
    # Initialize the density matrix.
    P=np.zeros((tb.statenum(Nrfl),tb.statenum(Nrfl)),dtype=complex)
    # Find out the shortest displacements of the whole lattice under periodic boundary condition.
    nptrs=ltc.periodictrsl(Nbl,1)
    # Read the initial density matrix.
    Pi=joblib.load(fileti)
    # Duplicate the matrix elements.
    for rid0 in range(Nrfl[0]):
        for rid1 in range(Nrfl[0]):
            # Keep only up to second-neighbor terms.
            if(NB[rid0,rid1]<=2):
                # Find out the shortest-distance equivalent site r1dm of r1 from r0.
                r1dm=ltc.pairdist(ltype,rs[rid0],rs[rid1],True,nptrs)[1][0]
                # Find out the id of r1dm in the initial lattice ris.
                r1dmcpid=ltc.siteid([[r1dm[0][n]%Nbli[n] for n in range(3)],r1dm[1]],ris)
                # Determine the pair density matrix of r0 and r1 from the initial density matrix.
                P01=tb.pairmat(Pi,rcpids[rid0],r1dmcpid,Nrfl[1])
                # Assign the pair density matrix.
                tb.setpair(P,P01,rid0,rid1,Nrfl[1])
    return P


def flrot(P,Nrfl,Ufl):
    '''
    Rotate the flavors of the density matrix.
    Return: A density matrix of the same size as P.
    P: Initial density matrix.
    Nrfl=[Nr,Nfl]: [site number, flavor number].
    Ufl: Unitary matrix for rotating the flavors.
    '''
    # Initialize a zero density matrix.
    Nst=tb.statenum(Nrfl)
    Pt=np.zeros((Nst,Nst),dtype=complex)
    # Add the rotated pair matrices to the density matrix.
    [tb.setpair(Pt,np.linalg.multi_dot([Ufl.conj().T,tb.pairmat(P,rid0,rid1,Nrfl[1]),Ufl]),rid0,rid1,Nrfl[1]) for rid0 in range(Nrfl[0]) for rid1 in range(Nrfl[0])]
    return Pt




'''Computation of orders.'''


def paircharge(P,rid0,rid1,Nfl,tobdg=False):
    '''
    Compute the charge of a pair of lattice sites. The onsite charge is real, while the offsite charge can be complex.
    '''
    return np.trace(tb.pairmat(P,rid0,rid1,Nfl,tobdg,0,0))


def pairspin(P,rid0,rid1,Nfl,tobdg=False):
    '''
    Compute the spin of a pair of lattice sites. The onsite spin is real, while the offsite spin can be complex.
    '''
    if(Nfl==2):smats=[tb.paulimat(n+1) for n in range(3)]
    elif(Nfl==4):smats=[tb.somat(0,n+1) for n in range(3)]
#    elif(Nfl==4):smats=[tb.somat(n+1,0) for n in range(3)]
    return np.array([np.trace(np.dot(tb.pairmat(P,rid0,rid1,Nfl,tobdg,0,0),(1./2.)*smats[n])) for n in range(3)])


def pairorbital(P,rid0,rid1,Nfl,tobdg=False):
    '''
    Compute the orbital of a pair of lattice sites. The onsite orbital is real, while the offsite orbital can be complex.
    '''
    if(Nfl==4):omats=[tb.somat(n+1,0) for n in range(3)]
    return np.array([np.trace(np.dot(tb.pairmat(P,rid0,rid1,Nfl,tobdg,0,0),(1./2.)*omats[n])) for n in range(3)])


def chargeorder(P,nb1ids,Nrfl,tobdg=False):
    '''
    Compute the charge order of the whole lattice. Return the lists of the site and bond orders and their maximal values.
    '''
    # Site order
    schs=[paircharge(P,rid,rid,Nrfl[1],tobdg).real for rid in range(Nrfl[0])]
    # Extract the order as the deviation from the average
    schsavg=sum(schs)/len(schs)
    schs=[sch-schsavg for sch in schs]
    schsa=[abs(sch) for sch in schs]
    schsmax=max(schsa)
    # Bond order
    bchs=[paircharge(P,pair[0],pair[1],Nrfl[1],tobdg) for pair in nb1ids]
    # Extract the order as the deviation from the average
    bchsavg=sum(bchs)/len(bchs)
    bchs=[bch-bchsavg for bch in bchs]
    # Distinguish the real and imaginary bonds
    bchsr=[bch.real for bch in bchs]
    bchsra=[abs(bchr) for bchr in bchsr]
    bchsi=[bch.imag for bch in bchs]
    bchsia=[abs(bchi) for bchi in bchsi]
    bchsrmax,bchsimax=max(bchsra),max(bchsia)

    return [[schs,bchsr,bchsi],[schsmax,bchsrmax,bchsimax]]


def spinorder(P,nb1ids,Nrfl,tobdg=False):
    '''
    Compute the spin order of the whole lattice. Return the lists of the site and bond orders and their maximal values.
    '''
    # Site order
    ssps=[pairspin(P,rid,rid,Nrfl[1],tobdg).real for rid in range(Nrfl[0])]
    sspsn=[np.linalg.norm(ssp) for ssp in ssps]
    sspsmax=max(sspsn)
    # Bond order
    bsps=[pairspin(P,pair[0],pair[1],Nrfl[1],tobdg) for pair in nb1ids]
    # Distinguish the real and imaginary bonds
    bspsr=[bsp.real for bsp in bsps]
    bspsi=[bsp.imag for bsp in bsps]
    bspsrn=[np.linalg.norm(bspr) for bspr in bspsr]
    bspsin=[np.linalg.norm(bspi) for bspi in bspsi]
    bspsrmax,bspsimax=max(bspsrn),max(bspsin)

    return [[ssps,bspsr,bspsi],[sspsmax,bspsrmax,bspsimax]]


def orbitalorder(P,nb1ids,Nrfl,tobdg=False):
    '''
    Compute the orbital order of the whole lattice. Return the lists of the site and bond orders and their maximal values.
    '''
    # Site order
    sobs=[pairorbital(P,rid,rid,Nrfl[1],tobdg).real for rid in range(Nrfl[0])]
    sobsn=[np.linalg.norm(sobs[nr]) for nr in range(len(sobs))]
    sobsmax=max(sobsn)
    # Bond order
    bobs=[pairorbital(P,pair[0],pair[1],Nrfl[1],tobdg) for pair in nb1ids]
    # Distinguish the real and imaginary bonds
    bobsr=[bobs[nb].real for nb in range(len(bobs))]
    bobsi=[bobs[nb].imag for nb in range(len(bobs))]
    bobsrn=[np.linalg.norm(bobsr[nb]) for nb in range(len(bobsr))]
    bobsin=[np.linalg.norm(bobsi[nb]) for nb in range(len(bobsi))]
    bobsrmax,bobsimax=max(bobsrn),max(bobsin)

    return [[sobs,bobsr,bobsi],[sobsmax,bobsrmax,bobsimax]]







