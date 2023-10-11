## Tight-binding square module

'''Tight-binding square module: Setup of tight-binding models on the square lattice'''

from math import *
import numpy as np

import sys
sys.path.append('../lattice')
import lattice as ltc
import tightbinding as tb




def spinlessbhz(t,lda,m,rs,nb1ids,Nbl,Nrfl,ltype,bc):
    '''
    Spinless Bernevig-Hughes-Zhang model: Realization of trivial and Chern insulators.
    H = sum_<ij> c_i^dagger ( -t*sigmaz + i*lda*(sigma.d_{ij}) ) c_j + sum_i c_i^dagger ( m*sigmaz ) c_i
    2-flavor fermions: c=[c_1,c_2]
    t: Flavor-dependent hopping
    lda: Spin-obit-type flavor-exchange hopping with displacement d_{ij}=r_i-r_j
    m: Flavor-dependent effective mass
    For t=lda=1: Trivial if m>4 or m<-4; chern if -4<m<0 and 0<m<4; Semimetallic if m=-4,0,4
    '''
    # Initial zero matrix
    H=np.zeros((tb.statenum(Nrfl),tb.statenum(Nrfl)),dtype=complex)
    # Onsite effective mass
    [tb.setpairpm(H,[0.,0.,0.,m],rid,rid,Nrfl[1]) for rid in range(Nrfl[0])]
    # Tight-binding hoppings 
    [tb.setpairpm(H,[0.,0.,0.,-t],pair[0],pair[1],Nrfl[1]) for pair in nb1ids]
    # Flavor-exchange hopping
    for pair in nb1ids:
        r0,r1=rs[pair[0]],rs[pair[1]]
        nptrs=ltc.periodictrsl(Nbl,bc)
        r1dm=ltc.pairdist(ltype,r0,r1,True,nptrs)[1][0]
        tb.setpairpm(H,[0.]+list(1.j*lda*(ltc.pos(r0,ltype)-ltc.pos(r1dm,ltype))),pair[0],pair[1],Nrfl[1])
    return H







