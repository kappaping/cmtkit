## Density-matrix module

'''Density-matrix module: Manipulations of density matrix.'''

from math import *
import numpy as np
from scipy.stats import unitary_group
import joblib
import time
import sparse

import sys
sys.path.append('../lattice')
import lattice as ltc
import brillouinzone as bz
import tightbinding as tb
import bogoliubovdegennes as bdg
import dmtriangular as dmtr
import dmkagome as dmka




'''Setup of density matrix'''


def projdenmat(U,n0,n1,Nst):
    '''
    Generate the density matrix by projecting on the n0-th to n1-th states of the unitary operator U=[u0,u1,u2,....].
    '''
    UT=U.conj().T
    # Project to only the Noc occupied states
    D=np.diag(np.array(n0*[0.]+(n1-n0)*[1.]+(Nst-n1)*[0.]))
    return np.round(np.linalg.multi_dot([U,D,UT]),25)


def setdenmat(Ptype,Nrfl,nf,fileti='',ltype='',rs=[],Nbl=[],NB=np.array([]),RDV=np.array([]),nbcpmax=-1,Nbli=[],toptb=False,ptb=0.01,toflrot=False,Ufl=np.array([[0.965926-0.12941j,-0.194114-0.112072j],[0.194114-0.112072j,0.965926+0.12941j]]),tobdg=False):
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
        if(tobdg==False):
            P=projdenmat(unitary_group.rvs(Nst),0,Noc,Nst)
        elif(tobdg):
            P=projdenmat(unitary_group.rvs(2*Nst),0,Nst,2*Nst)
            P+=np.block([[np.zeros((Nst,Nst)),np.zeros((Nst,Nst))],[np.zeros((Nst,Nst)),-np.identity(Nst)]])
            PBs=[[bdg.bdgblock(P,phid0,phid1) for phid1 in range(2)] for phid0 in range(2)]
            P00=(PBs[0][0]+(-PBs[1][1].T))/2.
            P01=(PBs[0][1]+PBs[1][0].conj().T)/2.
            P01=(P01+(-P01.T))/2.
            P=np.block([[P00,P01],[P01.conj().T,-P00.T]])
    # Read the density matrix from the file fileti.
    elif(Ptype=='read'):
        print('Read the density matrix from:', fileti)
        if(tobdg==False):P=joblib.load(fileti)
        elif(tobdg):P=joblib.load(fileti)[0]
    # Read the particle-hole density matrix from the file fileti and transform to a Bogoliubov-de Gennes form.
    elif(Ptype=='phtobdg'):
        print('Get a BdG density matrix by reading the particle-hole one from:', fileti)
        P00=joblib.load(fileti)
        P=bdg.phmattobdg(P00)
    # Copy the density matrix from the one in file fileti with size Nbli under periodic boundary condition.
    elif(Ptype=='copy'):
        print('Copy the density matrix with system size',Nbli,'from:', fileti)
        P=denmatcopy(ltype,rs,Nrfl,Nbl,NB,nbcpmax,fileti,Nbli,tobdg)
    # Others: The density matrix is assigned by other functions.
    else:
        print('Assign the density matrix as:', Ptype)
        # Add a uniform density distribution with filling nf to the density matrix.
        if(tobdg==False):P+=nf*np.identity(Nst)
        elif(tobdg):P+=nf*np.identity(2*Nst)
        dpe=0.1
        if(ltype=='tr'):dmtr.denmatans(P,Ptype,rs,NB,RDV,Nrfl,dpe,tobdg)
        elif(ltype=='ka'):dmka.denmatans(P,Ptype,rs,NB,RDV,Nrfl,dpe,tobdg)
    if(toptb):
        print('Perturb the density matrix at scale =',ptb)
        ptbr=ptb*np.max(np.abs(P))
        print('Perturbation rate =',ptbr)
        if(tobdg==False):P=(1.-ptbr)*P+ptbr*projdenmat(unitary_group.rvs(Nst),0,Noc,Nst)
        elif(tobdg):P=(1.-ptbr)*P+ptbr*projdenmat(unitary_group.rvs(2*Nst),0,Nst,2*Nst)
    if(toflrot):
        print('Rotate the flavors of the density matrix')
        P=flrot(P,Nrfl,Ufl,tobdg)

    return P


def denmatcopy(ltype,rs,Nrfl,Nbl,NB,nbcpmax,fileti,Nbli,tobdg=False):
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
    if(tobdg==False):P=np.zeros((tb.statenum(Nrfl),tb.statenum(Nrfl)),dtype=complex)
    elif(tobdg):P=np.zeros((2*tb.statenum(Nrfl),2*tb.statenum(Nrfl)),dtype=complex)
    # Find out the shortest displacements of the whole lattice under periodic boundary condition.
    nptrs=ltc.periodictrsl(Nbl,1)
    # Read the initial density matrix.
    if(tobdg==False):Pi=joblib.load(fileti)
    elif(tobdg):Pi=joblib.load(fileti)[0]
    if(nbcpmax==-1):
        if(max(Nbl)<=max(Nbli)):
            nbcpmax=round(np.max(NB))
            print('Copy all neighbors up to the',nbcpmax,'-th neighbors.')
        elif(max(Nbl)>max(Nbli)):
            nbcpmax=2
            print('Copy the neighbors up to the',nbcpmax,'-th neighbors.')
    else:
        print('Copy the neighbors up to the',nbcpmax,'-th neighbors.')
    # Duplicate the matrix elements.
    for rid0 in range(Nrfl[0]):
        for rid1 in range(Nrfl[0]):
            # Keep only up to second-neighbor terms.
            if(NB[rid0,rid1]<=nbcpmax):
                # Find out the shortest-distance equivalent site r1dm of r1 from r0.
                r1dm=ltc.pairdist(ltype,rs[rid0],rs[rid1],True,nptrs)[1][0]
                # Find out the id of r1dm in the initial lattice ris.
                r1dmcpid=ltc.siteid([[r1dm[0][n]%Nbli[n] for n in range(3)],r1dm[1]],ris)
                if(tobdg==False):
                    # Determine the pair density matrix of r0 and r1 from the initial density matrix.
                    Pi01=tb.pairmat(Pi,rcpids[rid0],r1dmcpid,Nrfl[1])
                    # Assign the pair density matrix.
                    tb.setpair(P,Pi01,rid0,rid1,Nrfl[1])
                elif(tobdg):
                    for phid0 in range(2):
                        for phid1 in range(2):
                            # Determine the pair density matrix of r0 and r1 from the initial density matrix.
                            Pi01=tb.pairmat(Pi,rcpids[rid0],r1dmcpid,Nrfl[1],tobdg=tobdg,phid0=phid0,phid1=phid1)
                            # Assign the pair density matrix.
                            tb.setpair(P,Pi01,rid0,rid1,Nrfl[1],tobdg=tobdg,phid0=phid0,phid1=phid1)
    return P


def flrot(P,Nrfl,Ufl,tobdg):
    '''
    Rotate the flavors of the density matrix.
    Return: A density matrix of the same size as P.
    P: Initial density matrix.
    Nrfl=[Nr,Nfl]: [site number, flavor number].
    Ufl: Unitary matrix for rotating the flavors.
    '''
    # Initialize a zero density matrix.
    Nst=tb.statenum(Nrfl)
    # Add the rotated pair matrices to the density matrix.
    if(tobdg==False):Pt=np.block([[np.linalg.multi_dot([Ufl.conj().T,tb.pairmat(P,rid0,rid1,Nrfl[1]),Ufl]) for rid1 in range(Nrfl[0])] for rid0 in range(Nrfl[0])])
    elif(tobdg):Pt=np.block([[np.block([[np.linalg.multi_dot([Ufl.conj().T,tb.pairmat(P,rid0,rid1,Nrfl[1],tobdg,phid0,phid1),Ufl]) for rid1 in range(Nrfl[0])] for rid0 in range(Nrfl[0])]) for phid1 in range(2)] for phid0 in range(2)])
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




'''Fourier transform'''


def momentumpairs(ks,bzop,qs,ltype,prds,chipm):
    '''
    Given the momenta ks and transfer momenta qs, list the momenta chipm*k+q for k in ks.
    Return: A matrix of ids of chipm*k=q.
    ks: Integral momenta.
    qs: Transfer momenta.
    ltype: Lattice type.
    prds: Periodicity of unit cell.
    chipm: 1 for particle-hole channel and -1 for particle-particle channel.
    '''
    print('Get the momentum pairs with chipm =',chipm)
    t0=time.time()
    # All high-symmetry points of the Brillouin zone.
    hsks=bz.hskpoints(ltype,prds)
    # Number of side pairs.
    Nsdp=round((len(hsks)-1)/2)
    # Edge centers of the Brillouin zone.
    kecs=[hsks[nsdp+1][1] for nsdp in range(Nsdp)]
    # Reciprocal lattice vectors.
    krls=np.array([2.*hsks[nkrl][1] for nkrl in [1,2]])
    Nk=len(ks)
    # Define the function which moves chipm*k+q into the Brillouin zone.
    def moveinbz(kq0,krls,kecs,Nsdp):
        kqs=[kq0+np.dot(np.array([sgn0,sgn1]),krls) for sgn0 in [0,-1,1] for sgn1 in [0,-1,1]]
        return kqs[np.argwhere(np.array([bz.inbz(kq,kecs,Nsdp,bzop=bzop) for kq in kqs]))[0,0]]
    # Get the list of chipm*k+q for k in ks and q in qs.
    kqids=np.array([[np.argwhere(np.array([np.linalg.norm(kq-kt)<1e-14 for kq in [moveinbz(chipm*k+q,krls,kecs,Nsdp)] for kt in ks]))[0,0] for k in ks] for q in qs])
    t1=time.time()
    print('Time for momentum pair =',t1-t0)
    return kqids


def symmeigenstates(Hk,ks,knkids,Nfl):
    print('Symmetrize the eigenstates under time-reversal symmetry.')
    Uees=[]
    def degeneracy(ees):
        dgns=[]
        for ee0 in ees:
            dgn=0
            for ee1 in ees:
                if(abs(ee0-ee1)<1e-14):dgn+=1
            dgns=dgns+[dgn]
        return max(dgns)
    for k in ks:
        ees,Uee=np.linalg.eigh(Hk(k))
        if(degeneracy(ees)>Nfl):
            if(np.linalg.norm(k)<1e-14):k=k+np.array([1e-5,0.,0.])
            else:kt=(1.-1e-5)*k
            ees,Uee=np.linalg.eigh(Hk(kt))
        Uees=Uees+[Uee]
    def timereversal(Uee):
        Ueet=Uee.conj()
        if(Nfl==2):Ueet=np.dot(np.tensordot(np.identity(ltc.slnum(ltype)),1.j*tb.paulimat(2),axes=0),Ueet)
        return Ueet
    for kid in range(len(ks)):
        Uees[knkids[kid]]=timereversal(Uees[kid])
    return Uees


def formfactor(P,Hk,ltype,rs,NB,RDV,Nrfl,ks,bzop,q,otype,nbds=[0,0],nbp=-1,tori='r',tobdg=False):
    if(otype=='c' or otype=='s' or otype=='o'):
        Pt=P
        if(tobdg):Pt=bdg.bdgblock(P,0,0)
        kpm=1
    elif(otype=='fe' or otype=='fo'):
        Pt=bdg.bdgblock(P,0,1)
        kpm=-1
    kqids=momentumpairs(ks,bzop,[q],ltype,[1,1,1],kpm)
    [Nr,Nfl]=Nrfl
    if(otype=='c'):orep=np.identity(Nfl)
    elif(otype=='s'):orep=tb.paulimat(3)
    elif(otype=='fe'):
        if(Nfl==1):orep=np.identity(1)
        elif(Nfl==2):orep=(tb.paulimat(0)+tb.paulimat(3))/2.
    elif(otype=='fo'):orep=1.j*tb.paulimat(2)
    Nk=len(ks)
    def fourierdenmat(kid):
        print(kid)
        ftidss=np.array([[tb.stateid(rid0,fl0,Nfl),tb.stateid(rid1,fl1,Nfl),tb.stateid(rs[rid0][1],fl0,Nfl),tb.stateid(rs[rid1][1],fl1,Nfl)] for rid0 in range(Nr) for fl0 in range(Nfl) for rid1 in range(Nr) for fl1 in range(Nfl)]).T
#        fts=np.array([orep[fl0,fl1]*(1./tb.statenum(Nrfl))*e**(-1.j*np.dot(ks[kid]+q/2,ltc.pos(rs[rid0],ltype))+kpm*1.j*np.dot(kpm*(ks[kid]+q/2)+q,ltc.pos(rs[rid0],ltype)-RDV[rid0,rid1])) for rid0 in range(Nr) for fl0 in range(Nfl) for rid1 in range(Nr) for fl1 in range(Nfl)])
        fts=np.array([orep[fl0,fl1]*(1./tb.statenum(Nrfl))*e**(-1.j*np.dot(ks[kid],ltc.pos(rs[rid0],ltype))+kpm*1.j*np.dot(ks[kqids[0,kid]],ltc.pos(rs[rid0],ltype)-RDV[rid0,rid1])) for rid0 in range(Nr) for fl0 in range(Nfl) for rid1 in range(Nr) for fl1 in range(Nfl)])
        return sparse.COO(ftidss,fts,shape=(tb.statenum(Nrfl),tb.statenum(Nrfl),tb.statenum([ltc.slnum(ltype),Nfl]),tb.statenum([ltc.slnum(ltype),Nfl])))
    Oks=np.array([sparse.tensordot(fourierdenmat(kid),Pt,axes=((0,1),(0,1)),return_type=np.ndarray) for kid in range(Nk)])
    if((otype=='fe' or otype=='fo') and np.linalg.norm(q)<1e-14):Uees=symmeigenstates(Hk,ks,kqids[0],Nfl)
    else:Uees=[np.linalg.eigh(Hk(k))[1] for k in ks]
    if(otype=='c' or otype=='s'):Oks=np.array([np.linalg.multi_dot([Uees[kid].conj().T,Oks[kid],Uees[kqids[0,kid]]]).round(12) for kid in range(Nk)])
    elif(otype=='fe' or otype=='fo'):Oks=np.array([np.linalg.multi_dot([Uees[kid].conj().T,Oks[kid],Uees[kqids[0,kid]].conj()]).round(12) for kid in range(Nk)])
    for kid in range(Nk):print('k =',ks[kid],', Ok =\n',Oks[kid])
#    oks=[np.linalg.svd(Ok) for Ok in Oks]
#    oks=[[ok[0].round(10),ok[1].round(10),ok[2].round(10)] for ok in oks]
#    for kid in range(Nk):print('k =',ks[kid],', ok =\n',oks[kid][0],'\n',oks[kid][1],'\n',oks[kid][2])
#    for kid in range(Nk):print('k =',ks[kid],', ok =\n',oks[kid][0],'\n',oks[kid][1],'\n',oks[kid][2])
#    oks=np.array([np.linalg.svd(Ok)[1] for Ok in Oks])
    oks=np.array([Ok[nbds[0],nbds[1]] for Ok in Oks])
    if(tori=='r'):
        print('Print real order.')
        oks=oks.real.tolist()
    elif(tori=='i'):
        print('Print imaginary order.')
        oks=oks.imag.tolist()
    return oks









