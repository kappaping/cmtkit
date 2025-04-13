## Hartree-Fock module

'''Hartree-Fock module: Functions of Hartree-Fock theory.'''

from math import *
import cmath as cmt
import numpy as np
import sparse
from scipy.optimize import fsolve
from scipy.optimize import fmin
from scipy.optimize import minimize
import joblib
import time

import sys
sys.path.append('../../cmt_code/lattice')
import lattice as ltc
sys.path.append('../../cmt_code/tightbinding')
import tightbinding as tb
import densitymatrix as dm
import bogoliubovdegennes as bdg




'''Matrix setup'''




def hfham(H0,P,UINT,tobdg=False,mu=0.):
    '''
    Hartree-Fock Hamiltonian: Including the noninteracting Hamiltonian and the Hartree-Fock potential.
    Return: A matrix Hhf of the same size as H0.
    H0: Noninteracting Hamiltonian.
    P: Density matrix.
    UINT: Interaction.
    Second term: Hartree potential UINT_{abcd}P_{cb}.
    Third term: Fock potential -UINT_{abdc}P_{cb}.
    '''
    if(tobdg==False):Hhf=H0+sparse.tensordot(UINT,P,axes=((1,2),(1,0)))-sparse.tensordot(UINT,P,axes=((1,3),(1,0)))
    elif(tobdg):
        PBs=[[bdg.bdgblock(P,phid0,phid1) for phid1 in range(2)] for phid0 in range(2)]
        Hhf00=sparse.tensordot(UINT,PBs[0][0],axes=((1,2),(1,0)))-sparse.tensordot(UINT,PBs[0][0],axes=((1,3),(1,0)))
        Hhf11=sparse.tensordot(UINT,PBs[1][1],axes=((2,1),(1,0))).T-sparse.tensordot(UINT,PBs[1][1],axes=((3,1),(1,0))).T
        Hhf01=sparse.tensordot(UINT,PBs[0][1],axes=((2,3),(1,0)))
        Hhf10=sparse.tensordot(UINT,PBs[1][0],axes=((0,1),(1,0)))
        Hhf=bdg.phmattobdg(H0,isham=True,mu=mu)+(1./2.)*np.block([[Hhf00,Hhf01],[Hhf10,Hhf11]])
    return Hhf


def energy(H0,P,Hhf,Nst,tobdg=False,mu=0.):
    '''
    Energy: Compute the state-averaged energy of density matrix P from the Hartree-Fock Hamiltonian Hhf.
    Return: Eenrgy ee=(1./2.)*Tr(P(H0+Hhf))/Nst averaged over the total number of states Nst.
    H0: Noninteracting Hamiltonian.
    P: Density matrix.
    Hhf: Hartree-Fock Hamiltonian.
    Nst: Total number of states.
    '''
    H0t,Nstt=H0,Nst
    if(tobdg):H0t,Nstt=bdg.phmattobdg(H0,isham=True,mu=mu),2*Nst
    ee=(1./2.)*np.trace(np.dot(P,H0t+Hhf)).real/Nstt
    return ee




'''Algorithm'''


def hartreefock(Pi,H0,UINT,NB,Nrfl,nf,tofile=False,filet='',optm=0,Nnb=1,printdm=20,writedm=200,Nhf=100000,Nhfm=0,dee0=1e-15,dp0=1e-15,Ptype='rand',tobdg=False,mu=0.,toexc=False):
    '''
    Hartree-Fock theory: Iterative algorithm for the computation of interacting ground states in the Hartree-Fock theory.
    Return: A NrxNr density matrix Pm for the ground state.
    Pi: Initial density matrix.
    H0: Noninteracting Hamiltonian.
    UINT: Interaction, a sparse rank-4 tensor.
    NB: NrxNr matrix of neighbor indices.
    Nrfl=[Nr,Nfl]: [site number, flavor number].
    nf: Filling fraction of each state.
    filet: File name for writing out the density matrix.
    optm: Additional optimization of the iterative algorithm.
        optm=0: None.
        optm=1: Optimal-damping algorithm (ODA).
    printdm: Periodicity of printing the computational progress.
    writedm: Periodicity of writing out the density matrix.
    Nhf: Maximal number of iterations.
    Nhfm: Minimal number of iterations.
    dee0: Cut-off error for the energy.
    dp0: Cut-off error for the density-matrix elements.
    toexc: Excite the top occupied state to the bottom unoccupied state.
    '''
    '''
    Initialization.
    '''
    Nst=tb.statenum(Nrfl)
    Noc=round(Nst*nf)
    nbidss=[ltc.nthneighbors(nnb,NB) for nnb in range(Nnb+1)]
    mut,mu1=mu,mu
    # Initial density matrix.
    Pm,Pmt=Pi,Pi
    dpm=1e-3
    dnfm1=1e-3
    if(tofile):
        print('Write density matrix to:',filet)
        joblib.dump(Pm,filet)
    # Initial Hartree-Fock Hamiltonian.
    Hhfm=hfham(H0,Pm,UINT,tobdg,mu)
    Hhfmt=Hhfm
    # Initial energy
    eem=energy(H0,Pm,Hhfm,Nst,tobdg,mu)
    # Initial printing: Energy and orders.
    printstatus(eem,Pm,nbidss,Nrfl,atit=0,tobdg=tobdg,mu=mu)
    # Set the added iteration number for convergence criteria.
    if(Ptype=='rand'):nmpcov=0
    else:nmpcov=1e4
    '''
    Iterative variation.
    '''
    for nm in range(Nhf):
        # Decide whether to print status and write out density matrix.
        toprint,towrite=(nm%printdm==0),(tofile and nm%writedm==0)
        # Get the density matrix.
        Pm1=getdenmat(Hhfmt,Nst,Noc,tobdg,toexc)
        # Setup the Hartree-Fock Hamiltonian.
        Hhfm1=hfham(H0,Pm1,UINT,tobdg,mu1)
        # Calculate the energy.
        eem1=energy(H0,Pm1,Hhfm1,Nst,tobdg,mu1)
        # Errors
        deem,dpm=eem1-eem,np.max(np.abs(Pm1-Pm))
        if(tobdg):
            dnfm1=bdg.denmatfilling(Pm1,Nst)-nf
            dnf0=max(dpm/(Nst*(nm+nmpcov+1))**2,1e-16)
            maxiter0=min(Nst,200)
        # Print out the status.
        if(toprint):printstatus(eem1,Pm1,nbidss,Nrfl,atit=1,n=nm,dee=deem,dp=dpm,tobdg=tobdg,mu=mu1,dnf=dnfm1)
        # Write out the density matrix and the chemical potential.
        if(tofile and towrite):
            if(tobdg):mu=getchempot(H0,Pm1,UINT,nf,Nst,mu1,tobdg=True,dnf0=dnf0,maxiter=maxiter0,toprint=toprint)
            writetofile(Pm1,filet,tobdg=tobdg,mu=mu)
        # Break if the error is small enough.
#        if(-dee0<deem<=0. and (eem1<eemmin or abs(deemmin)<dee0) and dpm<dp0 and nm>Nhfm):
        if(-dee0<deem<=0. and dpm<dp0 and nm>Nhfm):
            eem,Pm=eem1,Pm1
            if(tobdg):mu=getchempot(H0,Pm1,UINT,nf,Nst,mu1,tobdg=True,dnf0=dnf0,maxiter=maxiter0,toprint=toprint)
            break
        # Optimal-damping algorithm (ODA).
        Pm1t=Pm1
        if(optm==1):Pm1t,mut=oda(Pmt,Pm1,Hhfmt,Hhfm1,H0,toprint,tobdg,mut,mu1,nm,nmpcov,deem,dpm)
        # Print out the status of interpolated density matrix.
        if(toprint):printstatus(eem1,Pm1t,nbidss,Nrfl,tonoit=True,tobdg=tobdg)
        # Search for the chemical potential that fixes the filling at nf.
        if(tobdg):mu1=getchempot(H0,Pm1t,UINT,nf,Nst,mut,tobdg=True,dnf0=dnf0,maxiter=maxiter0,toprint=toprint)
        # Construct the new Hartree-Fock Hamiltonian.
        Hhfmt=hfham(H0,Pm1t,UINT,tobdg,mu1)
        # Reset the energy and density matrix for the next iteration.
        eem,Pm,Pmt=eem1,Pm1,Pm1t
        if(nm==Nhf-1):
            if(tobdg):mu=getchempot(H0,Pm1,UINT,nf,Nst,mu1,tobdg=True,dnf0=dnf0,maxiter=maxiter0,toprint=toprint)
    '''
    Ending.
    '''
    # Write out the density matrix
    if(tofile):writetofile(Pm,filet,tobdg=tobdg,mu=mu)
    # Print the final status
    printstatus(eem,Pm,nbidss,Nrfl,atit=2,n=nm,dee=deem,dp=dpm,tobdg=tobdg,mu=mu)
    return Pm


def writetofile(P,filet,tobdg=False,mu=0.):
    '''
    Write the density matrix to the file. If tobdg=True, also write the chemical potential mu.
    '''
    print('      Write to file.')
    if(tobdg==False):joblib.dump(P,filet)
    elif(tobdg):joblib.dump([P,mu],filet)


def printstatus(ee,P,nbidss,Nrfl,atit=1,n=-1,dee=0.,dp=0.,tonoit=False,tobdg=False,mu=0.,dnf=0.):
    '''
    Print out the status of the computation.
    ee: Current energy.
    P: Current density matrix.
    nb1ids: 1st-neighbor pairs of sites.
    Nrfl=[Nr,Nfl]: [site number, flavor number].
    atit: Position in the iteration. 0: Initial. 1: In-iteration. 2: End.
    n: Current number of iteration.
    dee: Energy error.
    dp: Density-martix element error.
    '''
    if(tonoit==False):
        if(atit==0):print('Initial: eem =',ee)
        elif(atit==1):
            print(n,'-th iteration: eem =',ee,'\n  deem =',dee,', dpm =',dp)
        elif(atit==2):
            print('End:',n,'-th iteration: eem =',ee,'\n  deem =',dee,', dpm =',dp)
        if(tobdg):print('  mu =',mu,', dnf =',dnf)
    # Orders
    print('     Order Maxs [0th neighbors,1st neighbors,....]:')
    odmaxss=dm.orders(P,nbidss,Nrfl,odtype='c',tobdg=tobdg)[1]
    print('     Ch: Rs =',odmaxss[0],', Is =',odmaxss[1])
    if(Nrfl[1]>1):
        odmaxss=dm.orders(P,nbidss,Nrfl,odtype='s',tobdg=tobdg)[1]
        print('     Sp: Rs =',odmaxss[0],', Is =',odmaxss[1])
    if(Nrfl[1]==4):
        ors=dm.orbitalorder(P,nbidss,Nrfl,tobdg=todbg)
        print('    som =',ors[1][0],', borm =',ors[1][1],', boim =',ors[1][2])
    if(tobdg):
        feps=bdg.flavorevenpairingorder(P,nbidss,Nrfl)
        print('    sfepm =',feps[1][0],', bfepm =',feps[1][1])
        if(Nrfl[1]>1):
            fops=bdg.flavoroddpairingorder(P,nbidss,Nrfl)
            print('    sfopm =',fops[1][0],', bfoprm =',fops[1][1])
    sys.stdout.flush()


def getdenmat(Hhf,Nst,Noc,tobdg=False,toexc=False):
    '''
    Get the density matrix from a Hartree-Fock Hamiltonian.
    Hhf: Hartree-Fock Hamiltonian.
    Nst: Total number of states.
    Noc: Number of occupied states.
    toexc: Excite the top occupied state to the bottom unoccupied state.
    '''
    #Diagonalize the Hartree-Fock Hamiltonian.
    U=np.linalg.eigh(Hhf)[1]
    # Assemble the density matrix by projecting to the lowest Noc eigenstates.
    if(tobdg==False):P=dm.projdenmat(U,0,Noc,Nst)
    elif(tobdg):
        P=dm.projdenmat(U,0,Nst,2*Nst)
        P+=np.block([[np.zeros((Nst,Nst)),np.zeros((Nst,Nst))],[np.zeros((Nst,Nst)),-np.identity(Nst)]])
        PBs=[[bdg.bdgblock(P,phid0,phid1) for phid1 in range(2)] for phid0 in range(2)]
        P00=(PBs[0][0]+(-PBs[1][1].T))/2.
        P01=(PBs[0][1]+PBs[1][0].conj().T)/2.
        P01=(P01+(-P01.T))/2.
        P=np.block([[P00,P01],[P01.conj().T,-P00.T]])
#        print('Check density matrix: max(P00-(-P11^T)) =',np.max(np.abs(PBs[0][0]-(-PBs[1][1].T))),', max(P01-(-P01^T)) =',np.max(np.abs(PBs[0][1]-(-PBs[0][1].T))))
    # If(toexc): Exciton, corresponding to D=[1,1,....,1,0,1,0,0,....]
    if(toexc):P=dm.projdenmat(U,0,Noct-1,Nstt)+dm.projdenmat(U,Noct,Noct+1,Nstt)
    return P


def oda(Pmt,Pm1,Hhfmt,Hhfm1,H0,toprint,tobdg,mut,mu1,nm,nmpcov,deem,dpm):
    '''
    Optimal damping algorithm (ODA): Choose an interpolated density matrix with minimal energy between Pm1 and Pmt.
    Pmt: Old density matrix from last iteration.
    Pm1: New density matrix.
    Hhfmt: Old Hartree-Fock Hamiltonian from last iteration.
    Hhfm1: New Hartree-Fock Hamiltonian.
    H0: Noninteracting Hamiltonian.
    toprint: Print am and bm if True.
    '''
    # Compute the deviations of density matrix and Hartree-Fock Hamiltonian from last iteration.
    dPm,dHhfm=Pm1-Pmt,Hhfm1-Hhfmt
    H0t,H01=H0,H0
    if(tobdg):H0t,H01=bdg.phmattobdg(H0,isham=True,mu=mut),bdg.phmattobdg(H0,isham=True,mu=mu1)
    dH0=H01-H0t
    am=(1./2.)*np.trace(np.dot(Pmt,dH0+dHhfm)+np.dot(dPm,H0t+Hhfmt)).real
    bm=np.trace(np.dot(dPm,dH0+dHhfm)).real
    def foda(ldam,am,bm):
        return am*ldam+(1./2.)*bm*ldam**2
    ldam=1.
    ldamp=-am/bm
    if(foda(ldamp,am,bm)<foda(1.,am,bm)):
        if(0.1<ldamp<1.):ldam=ldamp
        elif(0.<ldamp<0.1):ldam=0.1
    if(toprint==1):print('      am = ',am,', bm = ',bm,', ldam = ',ldam)
    deema=1./((nm+nmpcov+1)**2)
    dpma=max((0.1*np.max(np.abs(Pm1)))/((nm+nmpcov+1)**(3+tobdg)),1e-14)
    if(toprint==1):print('      deema = ',deema,', dpma = ',dpma)
    if(((deem>-1e-15 or abs(deem)>deema) and dpm>dpma) or (tobdg and dpm>0.01*sqrt(abs(deem)))):
        ldam=0.25
        if(toprint==1):print('      reset ldam = ',ldam)
    return Pmt+ldam*dPm,mut+ldam*(mu1-mut)




'''Bogoliubov-de Gennes'''


def chempottofilling(H0,P,UINT,mu,nf,Nst):
    '''
    Given a density matrix P and a chemical potential mu, compute the Hartree-Fock Hamiltinian, obtain the corresponding density matrix, and determine its filling deviation dnf from the assigned filling nf.
    Return: Filling deviation dnf.
    '''
    Noc=round(Nst*nf)
    Hhf=hfham(H0,P,UINT,tobdg=True,mu=mu)
    Pt=getdenmat(Hhf,Nst,Noc,tobdg=True)
    dnf=bdg.denmatfilling(Pt,Nst)-nf
    return dnf


def chempottoenergy(H0,P,UINT,mu,nf,Nst,dnf0):
    '''
    Given a density matrix P and a chemical potential mu, compute the Hartree-Fock Hamiltinian, obtain the corresponding density matrix, and determine its filling deviation dnf from the assigned filling nf.
    Return: Filling deviation dnf.
    '''
    Hhf=hfham(H0,P,UINT,tobdg=True,mu=mu)
    ee=energy(H0,P,Hhf,Nst,tobdg=True,mu=mu)
    dnf=chempottofilling(H0,P,UINT,mu,nf,Nst)
    if(abs(dnf)>dnf0):ee=1e3*(1.+abs(dnf))
    return ee


def getchempot(H0,P,UINT,nf,Nst,mu=0.,tobdg=False,dnf0=1e-15,tol0=1e-17,maxiter=None,toprint=False,toread=False,filet=''):
    '''
    Given a density matrix P, search for the chemical potential mu that generates a new density matrix with filling nf.
    '''
    '''
    def fillingdev(mut):
        return chempottofilling(H0,P,UINT,mut,nf,Nst)
    # Search the chemical potential by fsolve
    mu0=mu
    mu1=fsolve(fillingdev,mu0,xtol=1e-15,maxfev=100)[0]
    dnf=fillingdev(mu1)
    '''
    if(tobdg==False):
        Hhf=hfham(H0,P,UINT)
        ees=np.linalg.eigvalsh(Hhf)
        Noc=round(Nst*nf)
        mu1=(ees[Noc-1]+ees[Noc])/2.
    elif(tobdg):
        if(toread):
            mu1=joblib.load(filet)[1]
            if(toprint):print('    mu =',mu1)
            return mu1
        def fenergy(mut):
            return chempottoenergy(H0,P,UINT,mut,nf,Nst,dnf0)
        mu0=mu
        mu1=fmin(fenergy,mu0,xtol=tol0,ftol=tol0,maxiter=maxiter,disp=False)[0]
        dnf=chempottofilling(H0,P,UINT,mu1,nf,Nst)
#        dnf0=chempottofilling(H0,P,UINT,mu0,nf,Nst)
#        ee0=fenergy(mu0)
#        ee1=fenergy(mu1)
#        if(toprint):print('    mu0 =',mu0,', dnf =',dnf0)
        if(toprint):print('      mu =',mu1,', dnf =',dnf)
#        if(toprint):print('    ee0 =',ee0,', ee1 =',ee1)
    return mu1


