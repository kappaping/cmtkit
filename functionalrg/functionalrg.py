## Functional renormalization group module

'''Functional renormalization group (RG) module: Algorithm of functional RG.'''

from math import *
import cmath as cmt
import numpy as np
import sparse
from scipy.spatial.transform import Rotation
import joblib
import time

import sys
sys.path.append('../../cmt_code/lattice')
import lattice as ltc
import brillouinzone as bz
sys.path.append('../../cmt_code/tightbinding')
import tightbinding as tb
import densitymatrix as dm
sys.path.append('../../cmt_code/bandtheory')
import bandtheory as bdth
sys.path.append('../hartreefock')




'''Truncation tools.'''


def truncationsites(ltype,rs,Nbl,NB,nbmax,prds,Nfl):
    '''
    Get the sites for truncation, which are bounded by a maximal neighbor index nbmax.
    Return: Unit-cell sites rucs, truncated sites rtrcs, unit-cell ids of truncated sites rtrcucids, neighbor-index matrix NBtrc, truncation projector Vtrc.
    ltype: Lattice type.
    rs: Lattice sites.
    Nbl: Bravais lattice dimensions.
    NB: Nr*Nr matrix of neighbor indices.
    nbmax: Maximal neighbor index allowed under truncation.
    prds: Periodicity of unit cell.
    Nfl: Flavor number.
    '''
    # Get the unit cell and its dimensions.
    rucs,Nuc=bdth.ucsites(ltype,prds)[0:2]
    # Get the site ids of the unit-cell sites.
    rucrids=[ltc.siteid(ruc,rs) for ruc in rucs]
    # Get the pairs within maximal neighbor index from the unit cell.
    pairtrcs=[[rucrid,rid] for rucrid in rucrids for rid in np.argwhere(NB[rucrid]<=nbmax).flatten().tolist()]
    # List the unit-cell ids for all lattice sites.
    rrucids=np.array([bdth.ucsiteid(r,prds,rucs,Nuc) for r in rs])
    # Initialize the lists of truncated sites, their ids, and their unit-cell ids.
    rtrcs=rucs
    rtrcrids=rucrids
    rucrtrcids=[rid for rid in range(len(rucs))]
    rtrcrucids=rucrtrcids
    # Get the translation vectors under periodic boundary condition.
    nptrs=ltc.periodictrsl(Nbl,bc=1)
    # Get the truncated sites.
    for pairtrc in pairtrcs:
        if((pairtrc[1] in rtrcrids)==False):
            rtrcs=rtrcs+[ltc.pairdist(ltype,rs[pairtrc[0]],rs[pairtrc[1]],totrsl=True,nptrs=nptrs)[1][0]]
            rtrcrids=rtrcrids+[pairtrc[1]]
            rtrcrucids=rtrcrucids+[rrucids[pairtrc[1]]]
    # Get the projector from the lattice to the truncated sites.
    stidss=np.array([[tb.stateid(rtrcid,fl,Nfl),tb.stateid(rtrcrids[rtrcid],fl,Nfl)] for rtrcid in range(len(rtrcrids)) for fl in range(Nfl)]).T
    vtrcs=np.array(stidss.shape[1]*[1.])
    Vtrc=sparse.COO(stidss,vtrcs,shape=(tb.statenum([len(rtrcrids),Nfl]),tb.statenum([len(rs),Nfl])))
    # Get the neighbor-index matrix of truncated sites.
    NBtrc=ltc.ltcpairdist(ltype,rtrcs,Nbl,bc=0)[0]
    print('Finish the truncation sites, number of sites =',len(rtrcs))
    return rucs,rtrcs,rtrcrucids,NBtrc,Vtrc


def truncator(rs,rucs,NB,nbmax,Nfl,topjuc=True):
    '''
    Construct a truncator which truncates two axes by maximal neighbor index nbmax.
    Return: A rank-4 tensor V.
    rs: Lattice sites.
    rucs: Unit-cell sites.
    NB: Neighbor-index matrix.
    nbmax: Maximal neighbor index allowed under truncation.
    Nrfl=[Nr,Nfl]: Number of sites and flavors.
    topj0: True if the first axis should be projected to the unit cell.
    '''
    # If topjuc, get the pairs within maximal neighbor index from the unit cell.
    if(topjuc):
        rucrids=[ltc.siteid(ruc,rs) for ruc in rucs]
        pairtrcs=[[rucid,rid] for rucid in range(len(rucs)) for rid in np.argwhere(NB[rucrids[rucid]]<=nbmax).flatten().tolist()]
    # If topjuc=False, get the pairs within maximal neighbor index.
    elif(topjuc==False):
        pairtrcs=np.argwhere(NB<=nbmax).tolist()
    # Construct a sparse rank-4 tensor for the truncator. The identities are assigned only to the allowed site pairs.
    stidss=np.array([[tb.stateid(pair[0],fl0,Nfl),tb.stateid(pair[1],fl1,Nfl),tb.stateid(pair[0],fl0,Nfl),tb.stateid(pair[1],fl1,Nfl)] for pair in pairtrcs for fl0 in range(Nfl) for fl1 in range(Nfl)]).T
    vs=np.array(stidss.shape[1]*[1.])
    if(topjuc):
        Nucst=tb.statenum([len(rucs),Nfl])
        Nst=tb.statenum([len(rs),Nfl])
        V=sparse.COO(stidss,vs,shape=(Nucst,Nst,Nucst,Nst))
    elif(topjuc==False):
        Nst=tb.statenum([len(rs),Nfl])
        V=sparse.COO(stidss,vs,shape=(Nst,Nst,Nst,Nst))
    return V


def ucprojector(rs,rucs,Nfl):
    '''
    Construct a projector which projects to the unit cell.
    Return: A rank-2 tensor Vuc.
    rs: Lattice sites.
    rucs: Unit-cell sites.
    Nfl: Flavor number.
    '''
    # Get the site ids of the zero-th unit cell.
    rucrids=[ltc.siteid(ruc,rs) for ruc in rucs]
    # Construct a sparse rank-2 tensor for the truncator. The identities are assigned only to the zero-th unit cell.
    stidss=np.array([[tb.stateid(rucid,fl,Nfl),tb.stateid(rucrids[rucid],fl,Nfl)] for rucid in range(len(rucs)) for fl in range(Nfl)]).T
    vucs=np.array(stidss.shape[1]*[1.])
    Nucst=tb.statenum([len(rucs),Nfl])
    Nst=tb.statenum([len(rs),Nfl])
    Vuc=sparse.COO(stidss,vucs,shape=(Nucst,Nst))
    return Vuc




'''Symmetry boost.'''


def insbz(q,qcn,Nrot,Nmir):
    '''
    Determine if the transfer momentum is in a symmetry-determined sub Brillouin zone. The symmetries are C_Nrot and mirror.
    Return: True or False
    q: Transfer momentum.
    qcn: Normal vector along the center of the sub-Brillouin-zone boundary.
    Nrot: Number for C_Nrot rotation symmetry.
    Nmir: 1 if to apply mirror symmetry with respect to qcn.
    '''
    # The origin is in the sub Brillouin zone.
    if(np.linalg.norm(q)<1e-14):return True
    # Determine if the finite momentum is in the sub Brillouin zone.
    else:
        return (sin(-pi/Nrot)-1e-14<np.cross(qcn,q)[2]/np.linalg.norm(q)<(1-Nmir)*(sin(pi/Nrot)-1e-13)+1e-14)*(np.dot(q,qcn)/np.linalg.norm(q)>cos(pi/Nrot)-1e-14)


def listsbz(ltype,prds,Nrot,Nmir,qs,tobzsym=False):
    '''
    List the transfer-momentum points in a symmetry-determined sub Brillouin zone.
    Return: A list qis of the transfer momentum points in the sub Brillouin zone, and a normal vector along the center of the sub-Brillouin-zone boundary.
    ltype: Lattice type.
    prds: Periodicity of unit cell.
    Nrot: Number for C_Nrot rotation symmetry.
    Nmir: Number for mirror symmetry with respect to the center of sub-Brillouin-zone boundary.
    qs: Transfer-momentum points in the Brillouin zone.
    tobzsym: True if the symmetry boost is demanded.
    '''
    # If the symmetry boost is not demanded, return the original list of transfer momenta.
    if(tobzsym==False):
        print('No symmetry applied.')
        qis,qcn=qs,np.array([0.,0.,0.])
    # If the symmetry boost is demanded, find out all of the points in the sub Brillouin zone.
    elif(tobzsym):
        print('Apply the C_Nrot rotation symmetry with Nrot =',Nrot,'and with',Nmir,'mirror symmetry')
        # All high-symmetry points of the Brillouin zone.
        hsks=bz.hskpoints(ltype,prds)
        # Number of side pairs.
        Nsdp=round((len(hsks)-1)/2)
        # Edge centers of the Brillouin zone.
        kecs=[hsks[nsdp+1][1] for nsdp in range(Nsdp)]
        # Find the normal vector along the center of the sub-Brillouin-zone boundary.
        qcn=sum([(((-1)**(Nsdp-2))**np.sign(nsdp))*kecs[nsdp] for nsdp in range(Nsdp)])
        qcn=qcn/np.linalg.norm(qcn)
        # List the transfer momenta in the sub Brillouin zone.
        qis=[]
        for q in qs:
            if(insbz(q,qcn,Nrot,Nmir)):qis+=[q]
        print('Number of momentum points in the sub Brillouin zone =',len(qis))
    return qis,qcn


def rotatevec(x,qcno,nrot,Nrot,nmir):
    '''
    Transform the vector x by nrot C_Nrot rotation and nmir mirror.
    Return: A vector of the result.
    x: The vector to transform.
    qcno: The normal vector orthogonal to the center of sub-Brillouin-zone boundary.
    nrot: The number of times the C_Nrot rotation is applied.
    Nrot: Number for C_Nrot rotation symmetry.
    nmir: The number of times the mirror symmetry is applied with respect to the center of sub-Brillouin-zone boundary.
    '''
    return np.dot(Rotation.from_rotvec(nrot*(2*pi/Nrot)*np.array([0,0,1])).as_matrix(),x)-2*nmir*np.dot(qcno,np.dot(Rotation.from_rotvec(nrot*(2*pi/Nrot)*np.array([0,0,1])).as_matrix(),x))*qcno


def sbzids(qs,qis,qcn,Nrot,Nmir):
    '''
    List the sub-Brillouin-zone ids of all transfer-momentum points under C_Nrot-rotation and Nmir-mirror symmetries.
    Return: A vector of the result.
    qs: Transfer-momentum points in the Brillouin zone.
    qis: Transfer-momentum points in the sub Brillouin zone.
    qcn: The normal vector along the center of sub-Brillouin-zone boundary.
    Nrot: Number for C_Nrot rotation symmetry.
    Nmir: Number for mirror symmetry with respect to the center of sub-Brillouin-zone boundary.
    '''
    # Find the normal vector orthogonal to the center of sub-Brillouin-zone boundary.
    qcno=np.dot(Rotation.from_rotvec((pi/2)*np.array([0,0,1])).as_matrix(),qcn)
    # Define the function which transforms a momentum into the sub Brillouin zone.
    def moveinsbz(q):
        # List all of the results of transforming q under rotations and mirrors.
        qrs=np.array([[rotatevec(q,qcno,nrot,Nrot,nmir) for nmir in range(Nmir+1)] for nrot in range(Nrot)])
        # Find the indices of rotation and mirror which transform q into the sub Brillouin zone.
        nqr=np.argwhere(np.array([[insbz(qrs[nrot,nmir],qcn,Nrot,Nmir) for nmir in range(Nmir+1)] for nrot in range(Nrot)]))[0]
        return qrs[nqr[0],nqr[1]],nqr
    # Get the ids of all transfer momenta q in the sub Brillouin zone. Keep the transfer-momentum ids and the symmetry idices.
    qqiids=[]
    for q in qs:
        qr,nqr=moveinsbz(q)
        qqiid=np.argwhere(np.array([np.linalg.norm(qr-qi)<1e-14 for qi in qis]))[0,0]
        qqiids+=[[qqiid,nqr]]
    return qqiids


def rotatertrcs(rtrcs,rucs,ltype,Nrot,Nmir,qcn):
    '''
    List the sub-Brillouin-zone ids of all transfer-momentum points under C_Nrot-rotation and Nmir-mirror symmetries.
    Return: A vector of the result.
    qs: Transfer-momentum points in the Brillouin zone.
    qis: Transfer-momentum points in the sub Brillouin zone.
    qcn: The normal vector along the center of sub-Brillouin-zone boundary.
    Nrot: Number for C_Nrot rotation symmetry.
    Nmir: Number for mirror symmetry with respect to the center of sub-Brillouin-zone boundary.
    '''
    qcno=np.dot(Rotation.from_rotvec((pi/2)*np.array([0,0,1])).as_matrix(),qcn)
    rct=sum([ltc.pos(ruc,ltype) for ruc in rucs])/len(rucs)
    rtrcrots=np.array([[[rct+rotatevec(ltc.pos(rtrc,ltype)-rct,qcno,nrot,Nrot,nmir) for rtrc in rtrcs] for nmir in range(Nmir+1)] for nrot in range(Nrot)])
    rtrcrotids=np.array([[[np.argwhere(np.array([np.linalg.norm(rtrcrots[nrot,nmir,nrtrc]-ltc.pos(rtrc,ltype))<1e-14 for rtrc in rtrcs]))[0,0] for nrtrc in range(len(rtrcs))] for nmir in range(Nmir+1)] for nrot in range(Nrot)])
    return rtrcrotids


def mapvertextosbz(qs,qis,qcn,rtrcs,rucs,rtrcrucids,ltype,Nrot,Nmir):
    print('Get the operator for the mapping of vertices into the sub Brillouin zone.')
    t0=time.time()
    qqiids=sbzids(qs,qis,qcn,Nrot,Nmir)
    Nq,Nqi=len(qs),len(qis)
    Nrtrc,Nruc=len(rtrcs),len(rucs)
    # Get the truncation-site ids of the unit-cell sites.
    rucrtrcids=[ltc.siteid(ruc,rtrcs) for ruc in rucs]
    rtrcrotids=rotatertrcs(rtrcs,rucs,ltype,Nrot,Nmir,qcn)
    vqiidss=np.array([[nq,rucid0,rtrcid1,rucid2,rtrcid3,qqiids[nq][0],rtrcrucids[rtrcrotids[qqiids[nq][1][0],qqiids[nq][1][1],rucrtrcids[rucid0]]],rtrcrotids[qqiids[nq][1][0],qqiids[nq][1][1],rtrcid1],rtrcrucids[rtrcrotids[qqiids[nq][1][0],qqiids[nq][1][1],rucrtrcids[rucid2]]],rtrcrotids[qqiids[nq][1][0],qqiids[nq][1][1],rtrcid3]] for nq in range(Nq) for rucid0 in range(len(rucs)) for rtrcid1 in range(Nrtrc) for rucid2 in range(len(rucs)) for rtrcid3 in range(Nrtrc)]).T
    vqis=np.array(vqiidss.shape[1]*[1.])
    VQI=sparse.COO(vqiidss,vqis,shape=(Nq,Nruc,Nrtrc,Nruc,Nrtrc,Nqi,Nruc,Nrtrc,Nruc,Nrtrc))
    t1=time.time()
    print('Finish the vertex-sbz-mapping operator, time =',t1-t0)
    return VQI




'''Fourier-transform tools'''


def fouriersites(ltype,rucs,rtrcs,rtrcrucids):
    '''
    Get the sites for truncation, which are bounded by a maximal neighbor index nbmax.
    Return: Unit-cell sites rucs, truncated sites rtrcs, unit-cell ids of truncated sites rtrcucids, neighbor-index matrix NBtrc, truncation projector Vtrc.
    ltype: Lattice type.
    rs: Lattice sites.
    Nbl: Bravais lattice dimensions.
    NB: Nr*Nr matrix of neighbor indices.
    nbmax: Maximal neighbor index allowed under truncation.
    prds: Periodicity of unit cell.
    Nfl: Flavor number.
    '''
    # Initialize the lists of truncated sites, their ids, and their unit-cell ids.
    rfts=rtrcs
    rftrucids=rtrcrucids
    # Get the truncated sites.
    for nrtrc0 in range(len(rtrcs)):
        rtrc0=rtrcs[nrtrc0]
        ruc0=rucs[rtrcrucids[nrtrc0]]
        for nrtrc1 in range(len(rtrcs)):
            rtrc1=rtrcs[nrtrc1]
            rft=[rtrc0[0]+(rtrc1[0]-ruc0[0]),rtrc1[1]]
            if(type(ltc.siteid(rft,rfts))!=int):
                rfts=rfts+[rft]
                rftrucids=rftrucids+[rtrcrucids[nrtrc1]]
    # Get the neighbor-index matrix of truncated sites.
    NBft=ltc.ltcpairdist(ltype,rfts,Nbl=[],bc=0)[0]
    print('Finish the Fourier-transform sites, number of sites =',len(rfts))
    return rfts,rftrucids,NBft


def fouriervertex(qs,qis,qcn,ltype,rtrcs,rucs,rtrcrucids,NBtrc,nbmax,Nfl,fttype,tobzsym,Nrot,Nmir):
    '''
    Get the Fourier-transform operator in the computation of crossed terms.
    Return: Rank-9 sparse tensors of Fourier-transform operator and its complex conjugate.
    qs: Transfer momenta.
    ltype: Lattice type.
    rtrcs: Truncation sites.
    rucs: Unit-cell sites.
    rtrcrucids: Unit-cell ids of truncation sites.
    NBtrc: Neighbor-index matrix of truncation sites.
    nbmax: Maximal neighbor index allowed under truncation.
    Nfl: Flavor number.
    '''
    print('Get the Fourier-transform operator for the vertices, type =',fttype)
    print('Get the Fourier-transform operator without flavor indices.')
    t0=time.time()
    if(fttype=='ft'):rfts,rftrucids,NBft=fouriersites(ltype,rucs,rtrcs,rtrcrucids)
    elif(fttype=='ct'):rfts,rftrucids,NBft=rtrcs,rtrcrucids,NBtrc
    # Get the truncation-site ids of the unit-cell sites.
    rucrftids=[ltc.siteid(ruc,rfts) for ruc in rucs]
    # Get the sets of one unit-cell site and three truncation sites, where all of the site pairs are within the maximal neighbor index.
    Nrft=len(rfts)
    ridsst=[[rucrftid,rftid1,rftid2,rftid3] for rucrftid in rucrftids for rftid1 in range(Nrft) for rftid2 in range(Nrft) for rftid3 in range(Nrft)]
    ridss=[]
    if(fttype=='ct'):
        for rids in ridsst:
            if(np.prod(np.array([NBft[rid0,rid1]<=nbmax for rid0 in rids for rid1 in rids]))):ridss=ridss+[rids]
    elif(fttype=='ft'):
        for rids in ridsst:
            if(NBft[rids[0],rids[1]]<=nbmax and NBft[rids[2],rids[3]]<=nbmax):ridss=ridss+[rids]
    print('Number of site sets =',len(ridss))
    # Get the Fourier-transform operator which transforms between the transfer momenta and truncation sites.
    Nq=len(qs)
    Nruc=len(rucs)
    Nrtrc=len(rtrcs)
    ftridss=np.array([[rftrucids[rids[0]],rids[1],rids[2],rids[3],nq,rftrucids[rids[0]],ltc.siteid(rfts[rids[1]],rtrcs),rftrucids[rids[2]],ltc.siteid([rucs[rftrucids[rids[2]]][0]+(rfts[rids[3]][0]-rfts[rids[2]][0]),rfts[rids[3]][1]],rtrcs)] for nq in range(Nq) for rids in ridss]).T
    ftrs=np.array([(1./Nq)*e**(1.j*np.dot(qs[nq],ltc.pos(rfts[rids[0]],ltype)-ltc.pos(rfts[rids[2]],ltype))) for nq in range(Nq) for rids in ridss])
    FTR=sparse.COO(ftridss,ftrs,shape=(Nruc,Nrft,Nrft,Nrft,Nq,Nruc,Nrtrc,Nruc,Nrtrc))
    t1=time.time()
    print('Finish the Fourier transform operator without flavor indices, time =',t1-t0)
    if(tobzsym):
        VQI=mapvertextosbz(qs,qis,qcn,rtrcs,rucs,rtrcrucids,ltype,Nrot,Nmir)
        print('Transform into sub Brillouin zone.')
        t0=time.time()
        FTR=sparse.tensordot(FTR,VQI,axes=((4,5,6,7,8),(0,1,2,3,4)),return_type=sparse.COO)
        t1=time.time()
        print('Finish the transformation, time =',t1-t0)
    print('Add the flavor indices.')
    t0=time.time()
    Nftst=tb.statenum([Nrft,Nfl])
    Nucst=tb.statenum([Nruc,Nfl])
    Ntrcst=tb.statenum([Nrtrc,Nfl])
    Nqi=len(qis)
    ftridss=FTR.coords.T.tolist()
    ftrs=FTR.data.tolist()
    ftidss=np.array([[tb.stateid(ftrids[0],fl0,Nfl),tb.stateid(ftrids[1],fl1,Nfl),tb.stateid(ftrids[2],fl2,Nfl),tb.stateid(ftrids[3],fl3,Nfl),ftrids[4],tb.stateid(ftrids[5],fl0,Nfl),tb.stateid(ftrids[6],fl1,Nfl),tb.stateid(ftrids[7],fl2,Nfl),tb.stateid(ftrids[8],fl3,Nfl)] for ftrids in ftridss for fl0 in range(Nfl) for fl1 in range(Nfl) for fl2 in range(Nfl) for fl3 in range(Nfl)]).T
    fts=np.array([ftr for ftr in ftrs for fl0 in range(Nfl) for fl1 in range(Nfl) for fl2 in range(Nfl) for fl3 in range(Nfl)])
    FT=sparse.COO(ftidss,fts,shape=(Nucst,Nftst,Nftst,Nftst,Nqi,Nucst,Ntrcst,Nucst,Ntrcst))
    FT=FT.round(14)
    t1=time.time()
    print('Finish adding the flavor indices, time =',t1-t0)
    # Get the inverse of the Fourier-transform operator.
    print('Get the inverse Fourier-transform operator.')
    t0=time.time()
    ftiidss=np.array([[tb.stateid(rftrucids[rids[0]],fl0,Nfl),tb.stateid(rids[1],fl1,Nfl),tb.stateid(rids[2],fl2,Nfl),tb.stateid(rids[3],fl3,Nfl),nqi,tb.stateid(rftrucids[rids[0]],fl0,Nfl),tb.stateid(ltc.siteid(rfts[rids[1]],rtrcs),fl1,Nfl),tb.stateid(rftrucids[rids[2]],fl2,Nfl),tb.stateid(ltc.siteid([rucs[rftrucids[rids[2]]][0]+(rfts[rids[3]][0]-rfts[rids[2]][0]),rfts[rids[3]][1]],rtrcs),fl3,Nfl)] for nqi in range(Nqi) for rids in ridss for fl0 in range(Nfl) for fl1 in range(Nfl) for fl2 in range(Nfl) for fl3 in range(Nfl)]).T
    ftis=np.array([e**(-1.j*np.dot(qis[nqi],ltc.pos(rfts[rids[0]],ltype)-ltc.pos(rfts[rids[2]],ltype))) for nqi in range(Nqi) for rids in ridss for fl0 in range(Nfl) for fl1 in range(Nfl) for fl2 in range(Nfl) for fl3 in range(Nfl)])
    FTI=sparse.COO(ftiidss,ftis,shape=(Nucst,Nftst,Nftst,Nftst,Nqi,Nucst,Ntrcst,Nucst,Ntrcst))
    FTI=FTI.round(14)
    t1=time.time()
    print('Finish the inverse Fourier-transform operator, time =',t1-t0)
    return FT,FTI




'''Computation of bare susceptibilities.'''


def blochstates(H0,ltype,rs,NB,RDV,Nrfl,nf,prds,ks,rtrcs,rtrcrucids,Vuc):
    '''
    Compute the Bloch states and their energies from a tight-binding Hamiltonian H0.
    Return: Tensor Eee of relative energies from chemical potential, tensors Pee and Pee0 of Bloch-state density matrices with and without projection to unit cell, bandwidth W.
    H0: Tight-binding Hamiltonian.
    ltype: Lattice type.
    rs: Lattice sites.
    NB: Neighbor-index matrix.
    RDV: Matrix of displacement vectors between site pairs.
    Nrfl=[Nr,Nfl]: [site number, flavor number].
    nf: Filling per state.
    prds: Periodicity of unit cell.
    ks: Integral momenta.
    rtrcs: Truncation sites.
    rtrcrucids: Unit-cell indices of the truncated sites.
    Vuc: Projector to the unit cell.
    '''
    print('Compute the Bloch states.')
    t0=time.time()
    # Set the unit cell with periodicity prds.
    rucs,RUCRP=bdth.ftsites(ltype,rs,prds)
    # Get the momentum-space Hamiltonian.
    Hk=lambda k:bdth.ftham(k,H0,Nrfl,RDV,rucs,RUCRP)
    # Diagonalize the Hamiltonian at each momentum k.
    Hks=[Hk(k) for k in ks]
    eeUs=[np.linalg.eigh(Hk) for Hk in Hks]
    # Sort the energies and get the bandwidth.
    eesrs=np.sort(np.array([eeU[0] for eeU in eeUs]).flatten())
    W=np.max(eesrs)-np.min(eesrs)
    print('Bandwidth W =',W)
    # Given a filling nf, compute the relative energy from the chemical potential.
    print('Filling =',nf)
    Noc=round(len(eesrs)*nf)
    mu=(eesrs[Noc-1]+eesrs[Noc])/2.
    print('Chemical potential =',mu)
    # Get the tensor of energies.
    Neek=Hks[0].shape[0]
    Eee=np.array([eeU[0]-mu*np.array(Neek*[1.]) for eeU in eeUs])
    # Define the Fourier-transform operator from the unit cell to the truncated sites.
    Nk=len(ks)
    Nfl=Nrfl[1]
    def fourierbloch(k):
        stidss=np.array([[tb.stateid(rtrcid,fl,Nfl),tb.stateid(rtrcrucids[rtrcid],fl,Nfl)] for rtrcid in range(len(rtrcs)) for fl in range(Nfl)]).T
        fts=np.array([sqrt(1./Nk)*e**(1.j*np.dot(k,ltc.pos(rtrcs[rtrcid],ltype))) for rtrcid in range(len(rtrcs)) for fl in range(Nfl)])
        return sparse.COO(stidss,fts,shape=(tb.statenum([len(rtrcs),Nfl]),tb.statenum([len(rucs),Nfl])))
    # Get the tensor of eigenstate density matrices.
    Uees=[sparse.dot(fourierbloch(ks[nk]),eeUs[nk][1]) for nk in range(Nk)]
    Pee=np.array([[dm.projdenmat(Uee,nee,nee+1,Neek) for nee in range(Neek)] for Uee in Uees]).round(14)
    # Project Pee to the unit cell.
    Pee0=sparse.tensordot(Pee,Vuc,axes=(2,1),return_type=sparse.COO)
    Pee0=sparse.tensordot(Pee0,Vuc,axes=(2,1),return_type=sparse.COO)
    t1=time.time()
    print('Finish the Bloch states, time =',t1-t0)
    return Eee,Pee,Pee0,W,mu


def momentumpairs(ks,qs,ltype,prds,chipm):
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
        return kqs[np.argwhere(np.array([bz.inbz(kq,kecs,Nsdp,bzop=True) for kq in kqs]))[0,0]]
    # Get the list of chipm*k+q for k in ks and q in qs.
    kqids=np.array([[np.argwhere(np.array([np.linalg.norm(kq-kt)<1e-14 for kq in [moveinbz(chipm*k+q,krls,kecs,Nsdp)] for kt in ks]))[0,0] for k in ks] for q in qs])
    t1=time.time()
    print('Time for momentum pair =',t1-t0)
    return kqids


def adaptivegrids(ltype,prds,ks,Nkc,Ngmax):
    print('Adaptively add more points in the momentum grids.')
    t0=time.time()
    bztype=bz.typeofbz(ltype,prds)
    if(bztype=='rc'):Nkgc0=2
    elif(bztype=='hx'):Nkgc0=3
#    Nkgc=(Nkgc0**np.sign(Ngmax))*(2**((Ngmax>=2)*(Ngmax-1)))
    Nkgc=(Ngmax*Nkgc0)**np.sign(Ngmax)
    kg0s=bz.weightedgrids(ltype,prds,Nkgc)
    kg0s=[[kg0[0],(1./Nkc)*kg0[1],-(1./Nkc)*kg0[1]] for kg0 in kg0s]
    Nk=len(ks)
    kgss=[[[kg0[0],k+kg0[1],k+kg0[2]] for kg0 in kg0s] for k in ks]
    t1=time.time()
    print('Finish the adaptive addition of momentum points, time =',t1-t0)
    return kgss,Nkgc


def gridenergies(H0,ltype,rs,RDV,Nrfl,prds,ks,Nkc,Ngmax,mu):
    '''
    Compute the Bloch states and their energies from a tight-binding Hamiltonian H0.
    Return: Tensor Eee of relative energies from chemical potential, tensors Pee and Pee0 of Bloch-state density matrices with and without projection to unit cell, bandwidth W.
    H0: Tight-binding Hamiltonian.
    ltype: Lattice type.
    rs: Lattice sites.
    NB: Neighbor-index matrix.
    RDV: Matrix of displacement vectors between site pairs.
    Nrfl=[Nr,Nfl]: [site number, flavor number].
    nf: Filling per state.
    prds: Periodicity of unit cell.
    ks: Integral momenta.
    rtrcs: Truncation sites.
    rtrcrucids: Unit-cell indices of the truncated sites.
    Vuc: Projector to the unit cell.
    '''
    print('Compute the energies in the momentum grids.')
    t0=time.time()
    # Set the unit cell with periodicity prds.
    rucs,RUCRP=bdth.ftsites(ltype,rs,prds)
    # Get the momentum-space Hamiltonian.
    Hk=lambda k:bdth.ftham(k,H0,Nrfl,RDV,rucs,RUCRP)
    kgss,Nkgc=adaptivegrids(ltype,prds,ks,Nkc,Ngmax)
    eegss=[[[kg[0],np.linalg.eigvalsh(Hk(kg[1]))-mu,np.linalg.eigvalsh(Hk(kg[2]))-mu] for kg in kgs] for kgs in kgss]
    t1=time.time()
    print('Finish the energies in the momentum grids, time =',t1-t0)
    return eegss,Nkgc


def baresuscepelem(ee0,ee1,T,chipm):
    '''
    Compute the two-state element of particle-hole or particle-particle bare susceptibility.
    Return: bare susceptibility element chi01=-(f(ee0,T)-f(chipm*ee1,T))/(ee0-chipm*ee1), where f(ee,T)=1/(e**(ee/T)+1) is the Fermi function.
    ee0,ee1: Relative energies of two states from chemical potential.
    T: Temperature.
    chipm: 1 for particle-hole channel and -1 for particle-particle channel.
    '''
    # Compute the numerator and denominator of the element.
    dee=ee0-chipm*ee1
    dfee=tb.fermi(ee0,T)-tb.fermi(chipm*ee1,T)
    # If the denominator is large enough, use it for later computation.
    if(abs(dee)>1e-10):
        # If the numerator is too small, return 0.
        if(abs(dfee)<1e-14):return 0.
        # If the numerator is large enough, divide it by the denominator.
        else:return -dfee/dee
    # If the denominator is too small, return the differential value.
    else:
        # If the denominator over temperature is small enough, return the differential value.
        if(abs(ee0/(2.*T))<20.):return 1./(4.*T*(cosh(ee0/(2.*T)))**2)
        # If the denominator over temperature is too large, return 0.
        else:return 0.


def baresuscepmat(Eee,kqids,eegss,T,chipm):
    print('Compute the bare susceptibility matrix for chipm =',chipm)
    t0=time.time()
    # Compute the two-state elements of bare susceptibility.
    Nk,Neek=Eee.shape
    Nq=kqids.shape[0]
    Nkg=len(eegss[0])
    chimidss=np.array([[nq,kqids[nq,nk],nee0,nk,nee1] for nq in range(Nq) for nk in range(Nk) for nee0 in range(Neek) for nee1 in range(Neek)]).T
#    chis=np.array([Nk*(1-(kqids[nq,nk]==nk and nee0==nee1))*baresuscepelem(Eee[kqids[nq,nk],nee0],Eee[nk,nee1],T,chipm) for nq in range(Nq) for nk in range(Nk) for nee0 in range(Neek) for nee1 in range(Neek)]).round(14)
    chims=np.array([Nk*sum([eegss[nk][nkg][0]*baresuscepelem(eegss[kqids[nq,nk]][nkg][round((3-chipm)/2)][nee0],eegss[nk][nkg][1][nee1],T,chipm) for nkg in range(Nkg)]) for nq in range(Nq) for nk in range(Nk) for nee0 in range(Neek) for nee1 in range(Neek)]).round(14)
    CHIM=sparse.COO(chimidss,chims,shape=(Nq,Nk,Neek,Nk,Neek))
    t1=time.time()
    print('Finish the bare susceptibility matrix, time =',t1-t0)
    return CHIM


def baresuscep(ltype,CHIM,Pee,Pee0,qs,chipm,rucs,Nfl,V,toprinfo=False):
    '''
    Construct a tensor of bare particle-hole or particle-particle susceptibility.
    Return: A sparse tensor of the bare susceptibility.
    ltype: Lattice type.
    Eee: Tensor of relative energies from chemical potential.
    Pee: Sparse tensor of eigenstate density matrices.
    Pee0: Sparse tensor of eigenstate density matrices, projected to the unit cell.
    qs: Transfer momenta.
    kqids: A matrix of ids of chipm*k+q.
    T: Temperature.
    chipm: 1 for particle-hole channel and -1 for particle-particle channel.
    rucs: Unit-cell sites.
    Nfl: Flavor number.
    V: Truncator.
    toprinfo: True if printing the progress information is desired.
    '''
    print('Compute the bare susceptibility for chipm =',chipm)
    t0=time.time()
    # Contraction: CHI[0,1,2,3,4]=CHI[0,x0,x1,1,2]Pee0[x0,x1,3,4].
    CHI=sparse.tensordot(CHIM,Pee0,axes=((1,2),(0,1)),return_type=sparse.COO)
    # Contraction: CHI[0,1,2,3,4]=CHI[0,x2,x3,1,2]Pee[x2,x3,3,4]=(CHI[0,x0,x1,x2,x3]Pee0[x0,x1,1,2])Pee[x2,x3,3,4].
    CHI=sparse.tensordot(CHI,Pee,axes=((1,2),(0,1)),return_type=sparse.COO)
    if(toprinfo):print('fin 1')
    # Insert the factor e**(-iq(r0-r1)) between the 0th and 1st axes.
    Nq=len(qs)
    Nucst=Pee0.shape[2]
    eqidss=np.array([[nq,stid0,stid1,nq,stid0,stid1] for nq in range(Nq) for stid0 in range(Nucst) for stid1 in range(Nucst)]).T
    eqs=np.array([e**(-1.j*np.dot(qs[nq],ltc.pos(rucs[stid0//Nfl],ltype)-ltc.pos(rucs[stid1//Nfl],ltype))) for nq in range(Nq) for stid0 in range(Nucst) for stid1 in range(Nucst)])
    EQ=sparse.COO(eqidss,eqs,shape=(Nq,Nucst,Nucst,Nq,Nucst,Nucst))
    CHI=sparse.tensordot(EQ,CHI,axes=((3,4,5),(0,1,2)),return_type=sparse.COO)
    if(toprinfo):print('fin 2')
    # Truncation: Truncate the pairs [(1,4),(2,3)] for particle-hole channel, and [(1,3),(2,4)] for particle-particle channel.
    if(chipm==1):CHI=sparse.tensordot(CHI,V,axes=((1,4),(0,1)),return_type=sparse.COO)
    elif(chipm==-1):CHI=sparse.tensordot(CHI,V,axes=((1,3),(0,1)),return_type=sparse.COO)
    CHI=sparse.tensordot(CHI,V,axes=((1,2),(0,1)),return_type=sparse.COO)
    if(toprinfo):print('fin 3')
    CHI=CHI.round(10)
    t1=time.time()
    print('Finish the bare susceptibility, time =',t1-t0)
    return CHI




'''Computation of interaction.'''


def truncateint(UINT,ltype,rs,rtrcs,Nbl,NB,nbmax,prds,Nfl,Vtrc,qs,toflsym,FTPAS):
    '''
    Truncate the interaction UINT into P, C, and D channels.
    Return: Sparse rank-5 tensors for the truncated interactions in P, C, and D channels.
    UINT: Interaction.
    ltype: Lattice type.
    rs: Lattice sites.
    rtrcs: Truncation sites.
    Nbl: Bravais lattice dimensions.
    NB: Neighbor-index matrix.
    nbmax: Maximal neighbor index allowed under truncation.
    prds: Periodicity of unit cell.
    Nfl: Flavor number.
    Vtrc: Truncation projector.
    qs: Transfer momenta.
    '''
    print('Truncate the initial interaction into P, C, and D channels.')
    t0=time.time()
    # Get the unit cell and its dimensions.
    rucs,Nuc=bdth.ucsites(ltype,prds)[0:2]
    # List the unit-cell ids for all r in rs.
    rrucids=np.array([bdth.ucsiteid(r,prds,rucs,Nuc) for r in rs])
    # Get the site ids of the unit-cell sites.
    rucrids=[ltc.siteid(ruc,rs) for ruc in rucs]
    # Get the projector to the unit cell.
    Vuc=ucprojector(rs,rucs,Nfl)
    if(toflsym):UINTt=UINT
    else:
        print('Antisymmetrize the bare interaction.')
        UINTt=(UINT-sparse.moveaxis(UINT,(0,1,3,2),(0,1,2,3)))/2.
    # Initialize the list of interactions in the P, C, and D channels, where the 0th axis is projected to the unit cell.
    UINTs=[sparse.tensordot(Vuc,UINTt,axes=(1,0),return_type=sparse.COO) for nuint in range(3)]
    # Get the truncator with unit-cell projection.
    V=truncator(rs,rucs,NB,nbmax,Nfl)
    # Contraction: UINTt[0,1,2,3]=V[0,1,x0,x1](UINTt[x0,x1,2,3] or UINTt[x0,2,x1,3] or UINTt[x0,2,3,x1]) for P, C, or D channel.
    UINTs=[sparse.tensordot(V,UINTs[nuint],axes=((2,3),(0,nuint+1)),return_type=sparse.COO) for nuint in range(3)]
    # Get the truncator without unit-cell projection.
    V=truncator(rs,rucs,NB,nbmax,Nfl,topjuc=False)
    # Contraction: UINTt[0,1,2,3]=UINTt[0,1,x0,x1]V[x0,x1,2,3].
    UINTs=[sparse.tensordot(UINTt,V,axes=((2,3),(0,1)),return_type=sparse.COO) for UINTt in UINTs]
    # Get the Fourier-transform operator, which transforms an axis to the unit cell. The exponential factor is computed with respect to the 0th axis. The paired axis is transformed simultaneously.
    Nq=len(qs)
    Nucst=tb.statenum([len(rucs),Nfl])
    Nst=tb.statenum([len(rs),Nfl])
    ftidss=np.array([[nq,tb.stateid(rucid,fl0,Nfl),tb.stateid(rrucids[stid1//Nfl],stid1%Nfl,Nfl),tb.stateid(ltc.siteid([ltc.cyc(rtrc[0],Nbl,bc=1),rtrc[1]],rs),fl1,Nfl),tb.stateid(rucid,fl0,Nfl),stid1,tb.stateid(ltc.siteid([ltc.cyc(rs[stid1//Nfl][0]+(rtrc[0]-rucs[rrucids[stid1//Nfl]][0]),Nbl,bc=1),rtrc[1]],rs),fl1,Nfl)] for nq in range(Nq) for rucid in range(len(rucs)) for fl0 in range(Nfl) for stid1 in range(Nst) for rtrc in rtrcs for fl1 in range(Nfl)]).T
    fts=np.array([e**(-1.j*np.dot(qs[nq],ltc.pos(rucs[rucid],ltype)-ltc.pos(rs[stid1//Nfl],ltype))) for nq in range(Nq) for rucid in range(len(rucs)) for fl0 in range(Nfl) for stid1 in range(Nst) for rtrc in rtrcs for fl1 in range(Nfl)])
    FT=sparse.COO(ftidss,fts,shape=(Nq,Nucst,Nucst,Nst,Nucst,Nst,Nst))
    # Fourier transform the interactions: UINTt[0,1,2,3,4]=FT[0,1,2,3,x0,x1,x2]UINTt[x0,4,x2,x1]
    UINTs=[sparse.tensordot(FT,UINTt,axes=((4,5,6),(0,3,2)),return_type=sparse.COO) for UINTt in UINTs]
    # Project the axes to the truncation sites.
    UINTs=[sparse.tensordot(UINTt,Vtrc,axes=(4,1),return_type=sparse.COO) for UINTt in UINTs]
    UINTs=[sparse.tensordot(UINTt,Vtrc,axes=(3,1),return_type=sparse.COO) for UINTt in UINTs]
    # Rearrange the axes from (0,1,3,2,4) to (0,1,2,3,4).
    UINTs=[sparse.moveaxis(UINTt,(0,1,3,2,4),(0,1,2,3,4)) for UINTt in UINTs]
    UINTs=[UINTt.round(10) for UINTt in UINTs]
    if(toflsym==False):UINTs=antisymmetrize(UINTs,FTPAS,'UINTs')
    t1=time.time()
    print('Finish the initial truncated interactions in the P, C, and D channels, time =',t1-t0)
    return UINTs


def oneloopdiagram(CHI,UINT0,UINT1,IDQ):
    '''
    Contract the 1-loop diagram involving bare susceptibility CHI and interactions UINT0 and UINT1.
    Return: A rank-5 sparse tensor for the correction from this diagram.
    CHI: Bare susceptibility.
    UINT0,UINT1: Interactions.
    IDQ: Rank-3 identity in transfer momenta.
    '''
    # Contraction: PHI[0,1,2,3,4,5]=IDQ[0,1,x0]UINT0[x0,2,3,4,5].
    PHI=sparse.tensordot(IDQ,UINT0,axes=(2,0),return_type=sparse.COO)
    # Contraction: PHI[0,1,2,3,4]=PHI[0,x1,1,2,x2,x3]CHI[x1,x2,x3,3,4]=(IDQ[0,x1,x0]UINT0[x0,1,2,x2,x3])CHI[x1,x2,x3,3,4].
    PHI=sparse.tensordot(PHI,CHI,axes=((1,4,5),(0,1,2)),return_type=sparse.COO)
    # Contraction: PHI[0,1,2,3,4,5]=IDQ[0,1,x0]PHI[x0,2,3,4,5].
    PHI=sparse.tensordot(IDQ,PHI,axes=(2,0),return_type=sparse.COO)
    # Contraction: PHI[0,1,2,3,4]=PHI[0,x1,1,2,x2,x3]UINT1[x1,x2,x3,3,4]=(IDQ[0,x1,x0]PHI[x0,1,2,x2,x3])UINT1[x1,x2,x3,3,4].
    PHI=sparse.tensordot(PHI,UINT1,axes=((1,4,5),(0,1,2)),return_type=sparse.COO)
    return PHI


def oneloopcorrection(PHIs,CHIs,UINTs0,UINTs1,toflsym,Nfl,FTPAS):
    '''
    Compute the 1-loop correction of the interaction.
    Return: A tensor dUINT for the 1-loop correction of interaction.
    CHIs: Bare susceptibilities.
    UINT0,UINT1: Interactions.
    toprinfo: True if printing the progress information is desired.
    '''
    print('Compute the one-loop corrections.')
    # Get the rank-3 identity for transfer momenta.
    Nq=PHIs[0].shape[0]
    idqidss=np.array([[nq,nq,nq] for nq in range(Nq)]).T
    idqs=np.array(idqidss.shape[1]*[1.])
    IDQ=sparse.COO(idqidss,idqs,shape=(Nq,Nq,Nq))
    # P channel: Particle-particle pairing.
    t0=time.time()
    dPHI0=-oneloopdiagram(CHIs[1],UINTs0[0],UINTs1[0],IDQ)
    dPHI0=dPHI0.round(10)
    PHIs[0]=PHIs[0]+dPHI0
    t1=time.time()
    print('Finish P, time =',t1-t0)
    # C channel: Particle-hole crossed.
    t0=time.time()
    if(toflsym):dPHI1=oneloopdiagram(CHIs[0],UINTs0[1],UINTs1[1],IDQ)
    else:dPHI1=2.*oneloopdiagram(CHIs[0],UINTs0[1],UINTs1[1],IDQ)
    dPHI1=dPHI1.round(10)
    PHIs[1]=PHIs[1]+dPHI1
    t1=time.time()
    print('Finish C, time =',t1-t0)
    # D channel: Particle-hole direct.
    t0=time.time()
    if(toflsym):dPHI2=oneloopdiagram(CHIs[0],UINTs0[1],UINTs1[2],IDQ)+oneloopdiagram(CHIs[0],UINTs0[2],UINTs1[1],IDQ)-Nfl*oneloopdiagram(CHIs[0],UINTs0[2],UINTs1[2],IDQ)
    else:dPHI2=-2.*oneloopdiagram(CHIs[0],UINTs0[2],UINTs1[2],IDQ)
    dPHI2=dPHI2.round(10)
    PHIs[2]=PHIs[2]+dPHI2
    t1=time.time()
    print('Finish D, time =',t1-t0)
    # Check the antisymmetry.
    if(toflsym==False):PHIs=antisymmetrize(PHIs,FTPAS,'PHIs')
    # Compute the maximal change in each channel.
    dphimax=max([np.max(np.abs(np.array(dPHI0.data.tolist()+[0.]))),np.max(np.abs(np.array(dPHI1.data.tolist()+[0.]))),np.max(np.abs(np.array(dPHI2.data.tolist()+[0.])))])
    print('Finish the one-loop corrections.')
    return PHIs,dphimax


def projectint(UINT0s,PHIs,FT,FTI,toflsym,FTPAS):
    '''
    Computation of the crossed terms in the evolution of interactions in P, C, and D channels.
    Return: Rank-5 sparse tensors for the crossed terms in the P, C, and D channels.
    PHIs: Vertices.
    FT and FTI: Fourier-transform operator and its complex conjugate.
    '''
    print('Compute the projected interactions in the P, C, and D channels.')
    # Fourier transform the vertices from transfer momenta to truncation sites.
    print('Fourier transform the vertices into real space.')
    t0=time.time()
    PHIFTs=[sparse.tensordot(FT,PHI,axes=((4,5,6,7,8),(0,1,2,3,4)),return_type=sparse.COO).round(10) for PHI in PHIs]
    t1=time.time()
    print('Finish the Fourier transform, time =',t1-t0)
    if(toflsym==False):
        dPP=(PHIFTs[0]-(-sparse.moveaxis(PHIFTs[0],(0,1,3,2),(0,1,2,3)))).data
        if(dPP.shape[0]==0):dPP=0
        else:dPP=np.max(np.abs(dPP))
        dDC=(PHIFTs[2]-(-PHIFTs[1])).data
        if(dDC.shape[0]==0):dDC=0
        else:dDC=np.max(np.abs(dDC))
        print('Check antisymmetry: PhiftP-(-PhiftPt) =',dPP,', PhiftD-(-PhiftC) =',dDC)
    # Initialize the interactions as the bare ones.
    UINTs=[]
    # Compute the P-channel interaction. The crossed terms are computed under the Fourier transforms from truncation sites to transfer momenta.
    t0=time.time()
    UPC=sparse.tensordot(FTI,PHIFTs[1],axes=((0,1,2,3),(0,3,2,1)),return_type=sparse.COO)
    UPD=sparse.tensordot(FTI,PHIFTs[2],axes=((0,1,2,3),(0,3,1,2)),return_type=sparse.COO)
    UINTs=UINTs+[UINT0s[0]+PHIs[0]+UPC+UPD]
    t1=time.time()
    print('Finish P, time =',t1-t0)
    # Compute the C-channel interaction. The crossed terms are computed under the Fourier transforms from truncation sites to transfer momenta.
    t0=time.time()
    UCP=sparse.tensordot(FTI,PHIFTs[0],axes=((0,1,2,3),(0,3,2,1)),return_type=sparse.COO)
    UCD=sparse.tensordot(FTI,PHIFTs[2],axes=((0,1,2,3),(0,2,1,3)),return_type=sparse.COO)
    UINTs=UINTs+[UINT0s[1]+PHIs[1]+UCP+UCD]
    t1=time.time()
    print('Finish C, time =',t1-t0)
    # Compute the D-channel interaction. The crossed terms are computed under the Fourier transforms from truncation sites to transfer momenta.
    t0=time.time()
    UDP=sparse.tensordot(FTI,PHIFTs[0],axes=((0,1,2,3),(0,2,3,1)),return_type=sparse.COO)
    UDC=sparse.tensordot(FTI,PHIFTs[1],axes=((0,1,2,3),(0,2,1,3)),return_type=sparse.COO)
    UINTs=UINTs+[UINT0s[2]+PHIs[2]+UDP+UDC]
    t1=time.time()
    print('Finish D, time =',t1-t0)
    UINTs=[UINT.round(10) for UINT in UINTs]
    # Check the antisymmetry.
    if(toflsym==False):UINTs=antisymmetrize(UINTs,FTPAS,'UINTs')
    print('Finish the projected interactions.')
    return UINTs


def pantisym(qs,ltype,rucs,rtrcs,rtrcrucids,Nfl):
    Nq=len(qs)
    Nruc=len(rucs)
    Nrtrc=len(rtrcs)
    Nucst=tb.statenum([Nruc,Nfl])
    Ntrcst=tb.statenum([Nrtrc,Nfl])
    ftpasidss=np.array([[nq,tb.stateid(rucid,fl0,Nfl),tb.stateid(rtrcid,fl1,Nfl),nq,tb.stateid(rtrcrucids[rtrcid],fl1,Nfl),tb.stateid(ltc.siteid([rucs[rtrcrucids[rtrcid]][0]+(rucs[rucid][0]-rtrcs[rtrcid][0]),rucs[rucid][1]],rtrcs),fl0,Nfl)] for nq in range(Nq) for rucid in range(Nruc) for rtrcid in range(Nrtrc) for fl0 in range(Nfl) for fl1 in range(Nfl)]).T
    ftpass=np.array([e**(1.j*np.dot(qs[nq],ltc.pos(rucs[rucid],ltype)-ltc.pos(rtrcs[rtrcid],ltype))) for nq in range(Nq) for rucid in range(Nruc) for rtrcid in range(Nrtrc) for fl0 in range(Nfl) for fl1 in range(Nfl)])
    FTPAS=sparse.COO(ftpasidss,ftpass,shape=(Nq,Nucst,Ntrcst,Nq,Nucst,Ntrcst)).round(14)
    return FTPAS


def antisymmetrize(Os,FTPAS,osname):
    print('Reinforce the antisymmetry of',osname)
    t0=time.time()
    # Check the antisymmetry of Os.
    O0as=sparse.tensordot(FTPAS,Os[0],axes=((3,4,5),(0,3,4)),return_type=sparse.COO)
    O0as=sparse.moveaxis(O0as,(0,3,4,1,2),(0,1,2,3,4))
    d00=np.max(np.abs(np.array((Os[0]-(-O0as)).data.tolist()+[0.])))
    d21=np.max(np.abs(np.array((Os[2]-(-Os[1])).data.tolist()+[0.])))
    print('Check the antisymmetry: [0]-(-[0]as) =',d00,', [2]-(-[1]) =',d21)
    # Correct the antysymmetry of Os.
    Os[0]=((Os[0]+(-O0as))/2.).round(10)
    Os[2]=((Os[2]+(-Os[1]))/2.).round(10)
    Os[1]=-Os[2]
    # Check the corrected antisymmetry of Os.
    O0as=sparse.tensordot(FTPAS,Os[0],axes=((3,4,5),(0,3,4)),return_type=sparse.COO)
    O0as=sparse.moveaxis(O0as,(0,3,4,1,2),(0,1,2,3,4))
    d00=np.max(np.abs(np.array((Os[0]-(-O0as)).data.tolist()+[0.])))
    d21=np.max(np.abs(np.array((Os[2]-(-Os[1])).data.tolist()+[0.])))
    print('After correction: [0]-(-[0]as) =',d00,', [2]-(-[1]) =',d21)
    t1=time.time()
    print('Finish the antisymmetry of',osname,', time =',t1-t0)
    return Os


def checkfermistat(Os,osname,FT,rucs,Nfl):
    '''
    Check if the Fermi statistics has been applied properly by looking at the onsite terms under Fourier transform.
    Os: List of objects.
    osname: String of the name of Os.
    FT: Fourier-transform operator.
    rucs: Unit-cell sites.
    Nfl: Flavor number.
    '''
    # Fourier transform the vertices from transfer momenta to truncation sites.
    print('Check the fermi statistics of',osname)
    OFTs=[sparse.tensordot(FT,O,axes=((4,5,6,7,8),(0,1,2,3,4)),return_type=sparse.COO).round(10) for O in Os]
    for noft in range(len(OFTs)):
#        print(nxft,'-th =',np.array([[[[PHIFTs[nxft][tb.stateid(rucid,fl0,Nfl),tb.stateid(rucid,fl1,Nfl),tb.stateid(rucid,fl2,Nfl),tb.stateid(rucid,fl3,Nfl)] for rucid in range(len(rucs)) for fl3 in range(Nfl)] for fl2 in range(Nfl)] for fl1 in range(Nfl)] for fl0 in range(Nfl)]))
        print(noft,'-th =',np.array([[OFTs[noft][tb.stateid(rucid,fl,Nfl),tb.stateid(rucid,fl,Nfl),tb.stateid(rucid,fl,Nfl),tb.stateid(rucid,fl,Nfl)] for fl in range(Nfl)] for rucid in range(len(rucs))]))




'''Algorithm of functional renormalization group.'''


def printcomponents(PHI,xname,rtrcs,Nfl,qs):
    '''
    Print the components of the object PHI.
    '''
    print(rtrcs)
    def components(PHI,nq,rid0,rid1,rid2,rid3,Nfl):
        return np.array([[[[PHI[nq,tb.stateid(rid0,fl0,Nfl),tb.stateid(rid1,fl1,Nfl),tb.stateid(rid2,fl2,Nfl),tb.stateid(rid3,fl3,Nfl)] for fl3 in range(Nfl)] for fl2 in range(Nfl)] for fl1 in range(Nfl)] for fl0 in range(Nfl)])
    qa=np.array([np.linalg.norm(q) for q in qs])
    nq=np.argwhere(qa<1e-14)[0,0]
    print('qs[nq] = ',qs[nq])
    rid0=ltc.siteid([np.array([0,0,0]),0],rtrcs)
    print(xname,rid0,rid0,rid0,rid0,' =\n',components(PHI,nq,rid0,rid0,rid0,rid0,Nfl),'\n')
    rid0,rid1=ltc.siteid([np.array([0,0,0]),0],rtrcs),ltc.siteid([np.array([1,0,0]),0],rtrcs)
    print(xname,rid0,rid1,rid1,rid0,' =\n',components(PHI,nq,rid0,rid1,rid1,rid0,Nfl),'\n')
    print(xname,rid0,rid1,rid0,rid1,' =\n',components(PHI,nq,rid0,rid1,rid0,rid1,Nfl),'\n')
    print(xname,rid0,rid0,rid1,rid1,' =\n',components(PHI,nq,rid0,rid0,rid1,rid1,Nfl),'\n')
    rid0=ltc.siteid([np.array([1,-1,0]),0],rtrcs)
    print(xname,rid0,rid0,rid0,rid0,' =\n',components(PHI,nq,rid0,rid0,rid0,rid0,Nfl),'\n')


def functionalrg(UINT,H0,ltype,rs,Nbl,NB,RDV,Nrfl,nf,prds=[1,1,1],nbmax=2,Nqc=12,Nkc=12,Nt=10000,uct=10.,toflsym=True,toadap=True,Ngmax=2,tobzsym=False,Nrot=1,Nmir=0,toprinfo=False):
    '''
    Perform the functional RG.
    Return: Final interaction UINTc, critical temperature Tc, and the flow of the maximal elements.
    UINT: Initial interaction.
    H0: Tight-binding Hamiltonian.
    ltype: Lattice type.
    rs: Lattice sites.
    Nbl: Bravais lattice dimensions.
    NB: Nr*Nr matrix of neighbor indices.
    RDV: Displacement vector between each pair of sites.
    Nrfl=[Nr,Nfl]: Number of sites and flavors.
    nf: Filling per state.
    prds: Periodicity of unit cell.
    nbmax: Maximal neighbor index allowed under truncation.
    Nqc: Momentum cut for transfer momenta.
    Nkc: Momentum cut for integral momenta.
    Nt: Maximal number of temperature steps.
    uct: Terminating scale for interaction over spectrum width U/W.
    toflsym: True/False if the flavor-symmetric/asymmetric formalism is chosen.
    toadap: True/False if the adaptive integral is/isn't applied.
    Ngmax: Maximal number of discretization index for integral momentum grids.
    tobzsym: True/False if the symmetry reduction of computation into sub Brillouin zone is/isn't applied for the transfer momenta.
    Nrot: Rotation number for C_Nrot symmetry.
    Nmir: Mirror number for mirror symmetry.
    toprinfo: True if printing the progress information is desired.
    '''
    # Version of FRG formalism.
    if(toflsym):
        print('Apply the flavor-symmetric formalism.')
        Nfl1lc=Nrfl[1]
        Nfl=1
        Nrfl=[Nrfl[0],Nfl]
    else:
        print('Apply the flavor-asymmetric formalism.')
        Nfl=Nrfl[1]
        Nfl1lc=1
    print('Flavor number for modeling Nfl =',Nfl,', flavor number for one-loop correction Nfl1lc =',Nfl1lc)
    # Get the truncation sites.
    print('Maximal neighbor index =',nbmax)
    rucs,rtrcs,rtrcrucids,NBtrc,Vtrc=truncationsites(ltype,rs,Nbl,NB,nbmax,prds,Nfl)
    # Get the discretized transfer momenta.
    print('Get transfer momenta qs.')
    qs=bz.listbz(ltype,prds,Nqc,bzop=True)[0]
    # Get the discretized transfer momenta in the sub Brillouin zone under symmetry.
    qis,qcn=listsbz(ltype,prds,Nrot,Nmir,qs,tobzsym=tobzsym)
    # Get the discretized integral momenta.
    print('Get integral momenta ks.')
    ks=bz.listbz(ltype,prds,Nkc,bzop=True)[0]
    # Get the ids of chipm*k+q, where chipm is 1 for particle-hole channel and -1 for particle-particle channel.
    chipms=[1,-1]
    kqidss=[momentumpairs(ks,qis,ltype,prds,chipms[nchi]) for nchi in range(2)]
    # Get the neighbor truncator V and unit-cell projector Vuc.
    V=truncator(rtrcs,rucs,NBtrc,nbmax,Nfl)
    Vuc=ucprojector(rtrcs,rucs,Nfl)
    # Compute the Bloch states from H0.
    Eee,Pee,Pee0,W,mu=blochstates(H0,ltype,rs,NB,RDV,Nrfl,nf,prds,ks,rtrcs,rtrcrucids,Vuc)
    # If the momentum integral is adaptive, compute the energies of the additional points in the momentum grids.
    print('To adjust the densities of points in momentum grids adaptively =',toadap)
    eegss0,Nkgc0=gridenergies(H0,ltype,rs,RDV,Nrfl,prds,ks,Nkc,Ngmax,mu)
    eegss1,Nkgc1=gridenergies(H0,ltype,rs,RDV,Nrfl,prds,ks,Nkc,Ngmax+1,mu)
    # Define the initial temperature scale at the bandwidth.
    Tm,Tad=W,W
    # Compute the bare particle-hole and particle-particle susceptibilities.
    CHIMs=[baresuscepmat(Eee,kqidss[nchi],eegss0,Tm,chipms[nchi]) for nchi in range(2)]
    CHI0s=[baresuscep(ltype,CHIMs[nchi],Pee,Pee0,qis,chipms[nchi],rucs,Nfl,V,toprinfo=toprinfo) for nchi in range(2)]
    CHIms=CHI0s
    # Set the initial interaction.
    FTPAS=pantisym(qis,ltype,rucs,rtrcs,rtrcrucids,Nfl)
    UINT0s=truncateint(UINT,ltype,rs,rtrcs,Nbl,NB,nbmax,prds,Nfl,Vtrc,qis,toflsym,FTPAS)
    UINTms=UINT0s
    # Get the Fourier-transform operator FT.
    FT,FTI=fouriervertex(qs,qis,qcn,ltype,rtrcs,rucs,rtrcrucids,NBtrc,nbmax,Nfl,'ct',tobzsym,Nrot,Nmir)
    # Check the Fermi statistics of the interaction.
    if(toflsym==False):checkfermistat(UINT0s,'UINT0s',FT,rucs,Nfl)
    # Set the initial vertices.
    PHIs=[sparse.zeros(shape=UINT0s[0].shape) for nphi in range(3)]
    # Monitor the RG flow with the maximal vertex element phimaxm.
    phimaxms=[0.,0.,0.]
    print('Initial: T0 = W =',Tm,', Max phis/W =',phimaxms/W)
    # Collect the maximal interaction uintmaxm element for the plot of RG flow. 
    phimaxss=np.array([[[Tm,phimaxm]] for phimaxm in phimaxms])
    # Set the initial time step.
    dtm=Tm/20.
    tongp1=False
    for m in range(Nt):
        Tm1=Tm-dtm
        # Compute the bare particle-hole and particle-particle susceptibilities.
        CHIM0s=[baresuscepmat(Eee,kqidss[nchi],eegss0,Tm1,chipms[nchi]) for nchi in range(2)]
        CHIM1s=[baresuscepmat(Eee,kqidss[nchi],eegss1,Tm1,chipms[nchi]) for nchi in range(2)]
        dchingmaxs=[np.max(np.abs(np.array((CHIM1s[nchi]-CHIM0s[nchi]).data.tolist()+[0.]))) for nchi in range(2)]
        chingmaxs=[np.max(np.abs(CHIM1s[nchi].data)) for nchi in range(2)]
        dching=max([dchingmaxs[nchi]/chingmaxs[nchi] for nchi in range(2)])
        print('Bare susceptibility error =',dching)
        if(dching>1e-3):
            print('Increase the number of points in momentum grids.')
            tongp1=True
        Nkgcm=Nkgc1
        CHIm1s=[baresuscep(ltype,CHIM1s[nchi],Pee,Pee0,qis,chipms[nchi],rucs,Nfl,V,toprinfo=toprinfo) for nchi in range(2)]
        dCHIs=[CHIm1s[nchi]-CHIms[nchi] for nchi in range(2)]
        # Compute the renormalization of interaction by the 1-loop corrections.
        PHIs,dphimax=oneloopcorrection(PHIs,dCHIs,UINTms,UINTms,toflsym,Nfl1lc,FTPAS)
        UINTm1s=projectint(UINT0s,PHIs,FT,FTI,toflsym,FTPAS)
        if(toflsym==False):checkfermistat(UINTm1s,'UINTm1s',FT,rucs,Nfl)
        # Monitor the RG flow with the maximal vertex element phimaxms.
        chimaxms=[np.max(np.abs(CHI.data)) for CHI in CHIm1s]
        dchimax=max([np.max(np.abs(dCHI.data)) for dCHI in dCHIs])
        qchimaxs=[qis[CHI.coords[0,np.argmax(np.abs(CHI.data))]]/pi for CHI in CHIm1s]
        phimaxms=[np.max(np.abs(np.array(PHI.data.tolist()+[0.]))) for PHI in PHIs]
        qphimaxs=[qis[np.concatenate((PHI.coords,np.array([[0,0,0,0,0]]).T),axis=1)[0,np.argmax(np.abs(np.array(PHI.data.tolist()+[0.])))]]/pi for PHI in PHIs]
        # Adaptive adjustment of decreasing temperature step.
        dtm1=min((1./20.)*Tm1,dphimax/dtm)
        # Adaptive adjustment of momentum cut.
        if(toadap and tongp1):
            print('Compute the energies in the momentum grids.')
            Ngmax+=1
            eegss0,Nkgc0=eegss1,Nkgc1
            eegss1,Nkgc1=gridenergies(H0,ltype,rs,RDV,Nrfl,prds,ks,Nkc,Ngmax+1,mu)
            CHIM1s=[baresuscepmat(Eee,kqidss[nchi],eegss1,Tm1,chipms[nchi]) for nchi in range(2)]
            CHIm1s=[baresuscep(ltype,CHIM1s[nchi],Pee,Pee0,qis,chipms[nchi],rucs,Nfl,V,toprinfo=toprinfo) for nchi in range(2)]
            tongp1=False
        # Update the iterative parameters.
        Tm,CHIms,UINTms=Tm1,CHIm1s,UINTm1s
        phimaxss=np.concatenate((phimaxss,np.array([[[Tm,phimaxm]] for phimaxm in phimaxms])),axis=1)
        # Print the status of RG.
        print('\n',m,'-th iteration: T =',Tm,', dtm =',dtm,', Nkc =',Nkc,', Nkgcm =',Nkgcm,'\nmax chis = ',chimaxms,', qchimaxs =',qchimaxs,', dchimax/dtm =',dchimax/dtm,'\nmax phis/W = ',phimaxms/W,', qphimaxs =',qphimaxs,', dphimax/dtm =',dphimax/dtm,'\n')
        # If the renormalized interaction exceeds the threshold uct, break the iteration.
        if(Tm<1e-6 or np.max(phimaxms)/W>uct):break
        dtm=dtm1
    dCHIcs=[CHIms[nchi]-CHI0s[nchi] for nchi in range(2)]
    UINTcs,PHIcs,Tc=UINTms,PHIs,Tm
#    printcomponents(CHIsm[0],'CHIs0',rs,Nrfl[1])
#    printcomponents(CHIsm[1],'CHIs1',rs,Nrfl[1])
#    printcomponents(UINTm,'UINTm',rs,Nrfl[1])

    return UINTcs,PHIcs,dCHIcs,Tc,phimaxss




'''Instability analysis'''


def leadinginstability(UINTcs,dCHIcs,qis,ks,ltype,rs,Nbl,NB,nbmax,prds,Nfl,toflsym=True,chtype=1,toct=True,nwps=[0],cwps=[1.],tosetq=0,idps=[0,0],reim=0):
    if(toflsym):
        if(chtype==0):Uc,dCHIc=UINTcs[0],dCHIcs[1]
        elif(chtype==1):Uc,dCHIc=-UINTcs[1],dCHIcs[0]
        elif(chtype==2):Uc,dCHIc=Nfl*UINTcs[2]-UINTcs[1],dCHIcs[0]
        Nfl=1
    else:
        if(chtype==1):Uc,dCHIc=2.*UINTcs[2],dCHIcs[0]
        elif(chtype==-1):Uc,dCHIc=UINTcs[0],dCHIcs[1]
    PHI=Uc
    if(toct):
        # Get the rank-3 identity for transfer momenta.
        Nq=Uc.shape[0]
        idqidss=np.array([[nq,nq,nq] for nq in range(Nq)]).T
        idqs=np.array(idqidss.shape[1]*[1.])
        IDQ=sparse.COO(idqidss,idqs,shape=(Nq,Nq,Nq))
        # Contraction: PHI[0,1,2,3,4,5]=IDQ[0,1,x0]UINT0[x0,2,3,4,5].
        PHI=sparse.tensordot(IDQ,Uc,axes=(2,0),return_type=sparse.COO)
        # Contraction: PHI[0,1,2,3,4]=PHI[0,x1,1,2,x2,x3]CHI[x1,x2,x3,3,4]=(IDQ[0,x1,x0]UINT0[x0,1,2,x2,x3])CHI[x1,x2,x3,3,4].
        PHI=sparse.tensordot(PHI,dCHIc,axes=((1,4,5),(0,1,2)),return_type=sparse.COO)
    rucs,rtrcs,rtrcrucids,NBtrc,Vtrc=truncationsites(ltype,rs,Nbl,NB,nbmax,prds,Nfl)
    rucrtrcids=[ltc.siteid(ruc,rtrcs) for ruc in rucs]
    nqmax=PHI.coords[0,np.argmax(np.abs(PHI.data))]
    if(tosetq==1):nqmax=np.argwhere(np.array([np.linalg.norm(qi) for qi in qis])<1e-14)[0,0]
    elif(tosetq==2):nqmax=np.argwhere(np.array([np.linalg.norm(qi-np.array([np.max(np.array(qis)[:,0]),0.,0.])) for qi in qis])<1e-14)[0,0]
    qmax=qis[nqmax]/pi
    print('qmax =',qmax)
    PHIqmax=PHI[nqmax]
    Nucst,Ntrcst=PHIqmax.shape[0],PHIqmax.shape[1]
    PHIqmaxM=PHIqmax.reshape((Nucst*Ntrcst,Nucst*Ntrcst)).todense()
    print('PHIqmax is herm =',np.max(np.abs(PHIqmaxM-PHIqmaxM.conj().T)))
    WL,xs,WRT=np.linalg.svd(PHIqmaxM)
    wcsgns=[np.sign(np.dot(WRT[nw:nw+1,:].flatten(),WL[:,nw:nw+1].flatten()).real) for nw in range(len(xs))]
#    wcs=[[wcsgns[nw]*xs[nw].round(10),wcsgns[nw]*WL[:,nw:nw+1].flatten().round(10)] for nw in range(len(xs))]
    wcs=[[wcsgns[nw]*xs[nw].round(10),wcsgns[nw]*WRT[nw:nw+1,:].conj().flatten().round(10)] for nw in range(len(xs))]
    wcs=sorted(wcs,key=lambda x:x[0])
    print([wcs[nw][0] for nw in range(10)])
    print('The',nwps,'-th state with coefficients =',cwps)
    wc0=sum([cwps[nw]*wcs[nwps[nw]][1] for nw in range(len(nwps))]).reshape((Nucst,Ntrcst)).round(10)
    Nk=len(ks)
    stidss=np.array([[nk,tb.stateid(rucid0,fl0,Nfl),tb.stateid(rtrcrucids[rtrcid1],fl1,Nfl),tb.stateid(rucid0,fl0,Nfl),tb.stateid(rtrcid1,fl1,Nfl)] for nk in range(len(ks)) for rucid0 in range(len(rucs)) for fl0 in range(Nfl) for rtrcid1 in range(len(rtrcs)) for fl1 in range(Nfl)]).T
    fts=np.array([sqrt(1./Nk)*e**(-1.j*(np.dot(ks[nk],ltc.pos(rucs[rucid0],ltype)-ltc.pos(rtrcs[rtrcid1],ltype)))) for nk in range(len(ks)) for rucid0 in range(len(rucs)) for fl0 in range(Nfl) for rtrcid1 in range(len(rtrcs)) for fl1 in range(Nfl)])
    Ntrcst=tb.statenum([len(rtrcs),Nfl])
    Nucst=tb.statenum([len(rucs),Nfl])
    FT=sparse.COO(stidss,fts,shape=(Nk,Nucst,Nucst,Nucst,Ntrcst))
    wc0=sparse.tensordot(FT,wc0,axes=((3,4),(0,1)),return_type=np.ndarray).round(10)
#    print(wc0)
    if(reim==0):wks=np.array([wc0[nk,idps[0],idps[1]].real for nk in range(len(ks))])
    elif(reim==1):wks=np.array([wc0[nk,idps[0],idps[1]].imag for nk in range(len(ks))])
    return wks




'''Renormalized Hartree-Fock-Bogoliubov'''


def criticalinteraction(H0,NB,UINTc,Tc,Nrfl,rs,nf,nbmax=2,tocr=False,tocfl=0,totrc0=False,ltype='',Nbl=[],prds=[1,1,1],RDV=np.array([]),Nk=24):
    '''
    From the interaction UINTc at Tc, get the critical interaction UINT0 that matches the critical scaling at T0.
    Return: Critical interaction UINT0.
    H0: Hamiltonian.
    NB: NrxNr matrix of neighbor indices.
    UINTc: Interaction at critical temeprature Tc.
    Tc: Critical temperature.
    Nrfl=[Nr,Nfl]: Number of sites and flavors.
    rs: Lattice sites.
    nf: Filling per state.
    nbmax: Maximal neighbor index allowed under truncation.
    totrc: True if doing contractions only for the final elements with 0-th axis in the 0-th unit cell.
    ltype: Lattice type.
    Nbl: Bravais lattice dimensions.
    prds: Periodicity of unit cell.
    '''
    print('Maximal neighbor index =',nbmax)
    # Compute the eigenstates of H0.
    eemus,W,Pees=blochstates(H0,ltype,rs,NB,RDV,Nrfl,nf,prds,Nk,nbmax,totrc0=totrc0)
    # Define the temperature scales.
    T0=W
    # Compute the bare susceptibilities.
    print('Critical temperature Tc =',Tc)
    CHIs=baresuscep(eemus,T0,Tc,Pees,NB,Nrfl[1],nbmax,totrc0=totrc0,ltype=ltype,rs=rs,prds=prds)
    rid0,rid1=ltc.siteid([[0,0,0],0],rs),ltc.siteid([[0,0,0],0],rs)
    print('CHIs[0,',rs[rid0],rs[rid0],rs[rid1],rs[rid1],'] =\n',np.array([[[[CHIs[0,tb.stateid(rid0,fl0,Nrfl[1]),tb.stateid(rid0,fl1,Nrfl[1]),tb.stateid(rid1,fl2,Nrfl[1]),tb.stateid(rid1,fl3,Nrfl[1])] for fl3 in range(Nrfl[1])] for fl2 in range(Nrfl[1])] for fl1 in range(Nrfl[1])] for fl0 in range(Nrfl[1])]))
    rid0,rid1=ltc.siteid([[0,0,0],0],rs),ltc.siteid([[1,0,0],0],rs)
    print('CHIs[0,',rs[rid0],rs[rid0],rs[rid1],rs[rid1],'] =\n',np.array([[[[CHIs[0,tb.stateid(rid0,fl0,Nrfl[1]),tb.stateid(rid0,fl1,Nrfl[1]),tb.stateid(rid1,fl2,Nrfl[1]),tb.stateid(rid1,fl3,Nrfl[1])] for fl3 in range(Nrfl[1])] for fl2 in range(Nrfl[1])] for fl1 in range(Nrfl[1])] for fl0 in range(Nrfl[1])]))
    print('CHIs[0,',rs[rid0],rs[rid1],rs[rid0],rs[rid1],'] =\n',np.array([[[[CHIs[0,tb.stateid(rid0,fl0,Nrfl[1]),tb.stateid(rid1,fl1,Nrfl[1]),tb.stateid(rid0,fl2,Nrfl[1]),tb.stateid(rid1,fl3,Nrfl[1])] for fl3 in range(Nrfl[1])] for fl2 in range(Nrfl[1])] for fl1 in range(Nrfl[1])] for fl0 in range(Nrfl[1])]))
    print('CHIs[0,',rs[rid0],rs[rid1],rs[rid1],rs[rid0],'] =\n',np.array([[[[CHIs[0,tb.stateid(rid0,fl0,Nrfl[1]),tb.stateid(rid1,fl1,Nrfl[1]),tb.stateid(rid1,fl2,Nrfl[1]),tb.stateid(rid0,fl3,Nrfl[1])] for fl3 in range(Nrfl[1])] for fl2 in range(Nrfl[1])] for fl1 in range(Nrfl[1])] for fl0 in range(Nrfl[1])]))
    rid0,rid1=ltc.siteid([[0,0,0],0],rs),ltc.siteid([[1,1,0],0],rs)
    print('CHIs[0,',rs[rid0],rs[rid0],rs[rid1],rs[rid1],'] =\n',np.array([[[[CHIs[0,tb.stateid(rid0,fl0,Nrfl[1]),tb.stateid(rid0,fl1,Nrfl[1]),tb.stateid(rid1,fl2,Nrfl[1]),tb.stateid(rid1,fl3,Nrfl[1])] for fl3 in range(Nrfl[1])] for fl2 in range(Nrfl[1])] for fl1 in range(Nrfl[1])] for fl0 in range(Nrfl[1])]))
    rid0,rid1=ltc.siteid([[0,0,0],0],rs),ltc.siteid([[0,0,0],0],rs)
    print('CHIs[1,',rs[rid0],rs[rid0],rs[rid1],rs[rid1],'] =\n',np.array([[[[CHIs[1,tb.stateid(rid0,fl0,Nrfl[1]),tb.stateid(rid0,fl1,Nrfl[1]),tb.stateid(rid1,fl2,Nrfl[1]),tb.stateid(rid1,fl3,Nrfl[1])] for fl3 in range(Nrfl[1])] for fl2 in range(Nrfl[1])] for fl1 in range(Nrfl[1])] for fl0 in range(Nrfl[1])]))

    UINTct=UINTc
    # If totrc0, consider only the parts with 0-th axis in the 0-th unit cell.
    if(totrc0):UINTct=truncelem0(UINTc,[0],[1,2,3],ltype,rs,NB,Nrfl[1],prds,nbmax)
    # Get the indices and elements of the sparse tensor of critical interaction UINTct.
    stidss=UINTct.coords.T.tolist()
    uintcs=UINTct.data
    duints=[]
    # Regarding the nonzero elements of UINTct as a vector b, get the matrix A for (1+A)x=b with x being the nonzero elements of UINT0 to solve.
    for stids in stidss:
        # Projector corresponding to each element of b.
        Pstids=sparse.COO(np.array([stids]).T,[1.],shape=UINTc.shape)
        # If totrc0, translate the elements in the 0-th unit cell to the whole lattice.
        if(totrc0):Pstids=inttrsl(Pstids,ltype,rs,Nbl,Nrfl[1],prds)
        # Compute the projected 1-loop correction.
        dUINT=oneloopcorrection(CHIs,Pstids,UINTc,NB,Nrfl[1],nbmax,tocr=tocr,tocfl=tocfl,totrc0=totrc0,ltype=ltype,rs=rs,Nbl=Nbl,prds=prds,totrsl=False)
        # Get the Column vectors of A.
        duint=np.array([dUINT[stids[0],stids[1],stids[2],stids[3]] for stids in stidss])
        print(stids,',len =',duint.shape)
        duints+=[duint]
    # Get the matrix 1+A.
    uint1beta=np.identity(len(uintcs))+np.array(duints).T
    # Solve the linear equation (1+A)x=b.
    uintc0s=np.linalg.solve(uint1beta,uintcs)
    # Get the critical interaction.
    UINTc0=sparse.COO(UINTct.coords,uintc0s,shape=UINTc.shape)
    # If totrc0, translate the result to the whole lattice.
    if(totrc0):UINTc0=inttrsl(UINTc0,ltype,rs,Nbl,Nrfl[1],prds)
    UINTc0=UINTc0.round(10)
    print('fin duint =',np.max(np.abs((UINTc0-(UINTc-oneloopcorrection(CHIs,UINTc0,UINTc,NB,Nrfl[1],nbmax,tocr=tocr,tocfl=tocfl,totrc0=totrc0,ltype=ltype,rs=rs,Nbl=Nbl,prds=prds))).data)))
    return UINTc0



