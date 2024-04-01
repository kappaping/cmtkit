## Interaction module

'''Interaction module: Functions of interaction.'''

from math import *
import cmath as cmt
import numpy as np
import sparse

import sys
sys.path.append('../../cmt_code/lattice')
import lattice as ltc
sys.path.append('../../cmt_code/tightbinding')
import tightbinding as tb




'''Interaction setup'''




def interaction(NB,Nrfl,us,utype='hu',RD=np.array([]),ldar=0.2,rob=1.,jhes=[0.],jcps=[1,2,3]):
    '''
    Interaction: Define the sparse rank-4 tensor for the interaction.
    Return: A sparse rank-4 tensor UINT of the interaction.
    NB,RD: NrxNr matrix of neighbor indices and distances.
    Nrfl=[Nr,Nfl]: [site number, flavor number].
    us: Density-density interactions [u0,u1,u2,....].
    utype: Type of density-density interaction, 'hu' for Hubbard and 'co' for Coulomb.
    ldar: Screening length of Coulomb repulsion.
    rob: Reduced ratio of interorbital interactions.
    jhes: Heisenberg exchange couplings [jhe0,jhe1,jhe2,....].
    '''
    # Density-density interaction.
    stids,uints=dendenint(NB,Nrfl,us,utype,RD,ldar,rob)
    # Sparse rank-4 tensor of interaction.
    Nst=tb.statenum(Nrfl)
    UINT=sparse.COO(stids,uints,shape=(Nst,Nst,Nst,Nst))
    # Heisenberg exchange.
    if(np.max(np.abs(jhes))>1e-12):
        stids,uints=heisenberg(NB,Nrfl,jhes,jcps=jcps)
        UINT=UINT+sparse.COO(stids,uints,shape=(Nst,Nst,Nst,Nst))
    return UINT


def dendenint(NB,Nrfl,us,utype,RD,ldar,rob):
    '''
    Density-density interaction: Define the density-density interactions.
    Return: State ids stids and values uints of nonzero elements in the sparse rank-4 interaction tensor.
    NB,RD: NrxNr matrix of neighbor indices and distances.
    Nrfl=[Nr,Nfl]: [site number, flavor number].
    us: Density-density interactions [u0,u1,u2,....].
    utype: Type of density-density interaction, 'hu' for Hubbard and 'co' for Coulomb.
        'hu': Assign the Hubbard interactions from the input us=[u0,u1,u2,....]
        'co': Assign the screened Coulomb interaction u(r)=(u0/sqrt((r/ldar)**2+1))*exp(-r/ldar) with screening length ldar.
    rob: Reduced ratio of interorbital interactions.
    '''
    print('Density-density interaction:')
    if(abs(rob-1.)>1e-12):print('Reduced ratio of interorbital interactions = ',rob)
    # Hubbard-type interaction.
    if(utype=='hu'):
        print('Hubbard interaction: [U0,U1,U2,....] =',us)
        # List the pairs of sites relevant to interactions, with the interactions assigned by neighboring distances.
        pairs=[pair for nb in range(len(us)) for pair in ltc.nthneighbors(nb,NB)]
        upairs=[us[NB[pair[0],pair[1]]] for pair in pairs]
    # Coulomb repulsion.
    elif(utype=='co'):
        print('Coulomb repulsion: U0 =',us[0],', ldar =',ldar)
        # List all pairs of sites, with the Coulomb repulsions assigned by neighboring distances.
        pairs=[[rid0,rid1] for rid0 in range(Nrfl[0]) for rid1 in range(Nrfl[0])]
        upairs=[(us[0]/sqrt((RD[pair[0],pair[1]]/ldar)**2+1))*exp(-RD[pair[0],pair[1]]/ldar) for pair in pairs]
    # Construct the state ids stids and values uints of nonzero elements in the sparse rank-4 interaction tensor.
    stids=np.array([[tb.stateid(pair[0],fl0,Nrfl[1]),tb.stateid(pair[1],fl1,Nrfl[1]),tb.stateid(pair[1],fl1,Nrfl[1]),tb.stateid(pair[0],fl0,Nrfl[1])] for pair in pairs for fl0 in range(Nrfl[1]) for fl1 in range(Nrfl[1])]).T.tolist()
    uints=[upairs[npr]*(((fl0//2)==(fl1//2))+rob*((fl0//2)!=(fl1//2))) for npr in range(len(pairs)) for fl0 in range(Nrfl[1]) for fl1 in range(Nrfl[1])]
    return stids,uints


def heisenberg(NB,Nrfl,jhes,jcps=[1,2,3]):
    '''
    Heisenberg exchange: Define the Heisenberg exchange.
    Return: State ids stids and values uints of nonzero elements in the sparse rank-4 interaction tensor.
    NB: NrxNr matrix of neighbor indices.
    Nrfl=[Nr,Nfl]: [site number, flavor number].
    jhes: Heisenberg exchange couplings [jhe0,jhe1,jhe2,....].
    '''
    print('Heisenberg exchange: [Jhe0,Jhe1,Jhe2,....] =',jhes,', components =',jcps)
    # List the pairs of sites relevant to interactions, with the interactions assigned by neighboring distances.
    pairs=[pair for nb in range(len(jhes)) for pair in ltc.nthneighbors(nb,NB)]
    upairs=[jhes[NB[pair[0],pair[1]]] for pair in pairs]
    # Construct the state ids stids and values uints of nonzero elements in the sparse rank-4 interaction tensor.
    stids=np.array([[tb.stateid(pair[0],fl0,Nrfl[1]),tb.stateid(pair[1],fl2,Nrfl[1]),tb.stateid(pair[1],fl3,Nrfl[1]),tb.stateid(pair[0],fl1,Nrfl[1])] for pair in pairs for fl0 in range(Nrfl[1]) for fl1 in range(Nrfl[1]) for fl2 in range(Nrfl[1]) for fl3 in range(Nrfl[1])]).T.tolist()
    uints=[(upairs[npr]/4.)*sum([tb.paulimat(jcp)[fl0,fl1]*tb.paulimat(jcp)[fl2,fl3] for jcp in jcps]) for npr in range(len(pairs)) for fl0 in range(Nrfl[1]) for fl1 in range(Nrfl[1]) for fl2 in range(Nrfl[1]) for fl3 in range(Nrfl[1])]
#    uints=[(upairs[npr]/4.)*(sum([tb.paulimat(comp)[fl0,fl1]*tb.paulimat(comp)[fl2,fl3] for comp in comps])-(fl0==fl1)*(fl2==fl3)) for npr in range(len(pairs)) for fl0 in range(Nrfl[1]) for fl1 in range(Nrfl[1]) for fl2 in range(Nrfl[1]) for fl3 in range(Nrfl[1])]
    return stids,uints




