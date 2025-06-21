## Monte Carlo module

'''Monte Carlo module: Functions of Monte Carlo'''

import numpy as np
import random

import sys
sys.path.append('../')
import lattice.lattice as ltc




def normalized(vt):
    '''
    Normalize a given vector
    '''
    nmvt=np.linalg.norm(vt)
    return (1./nmvt)*vt


def zfl(fltype,pm):
    '''
    Generate a Sz flavor
    '''
    dictt={
            1:np.array([pm]),              # Ising
            3:pm*np.array([0.,0.,1.])/2.    # Heisenberg
            }
    return dictt[fltype]


def randomfl(fltype):
    '''
    Generate a random flavor
    '''
    dictt={
            1:np.array([random.choice([-1,1])]),              # Ising
            3:normalized(np.array([random.uniform(-1.,1.)/2. for n in range(3)]))    # Heisenberg
            }
    return dictt[fltype]


def latticefl(rs,Nltc,fltype,tofm):
    # Random
    if(tofm==0):
        return [randomfl(fltype) for r in rs]
    # Ferromagnetism
    elif(tofm==1):
        return [zfl(fltype,1) for r in rs]
    # Antiferromagnetism
    elif(tofm==-1):
        return [zfl(fltype,1-2*((r[0][0]-r[0][1])%2)) for r in rs]


def flipsite(flr,fltype):
    '''
    Flip the flavor on a site
    '''
    dictt={
            1:-flr,       # Ising
            3:randomfl(3) # Heisenberg
            }
    return dictt[fltype]


def energyr(flr,fls,Js,pairsr,Nltc):
    return sum(Js[nd]*np.dot(flr,fls[ltc.rid(pairt[1],Nltc)]) for nd in range(len(pairsr)) for pairt in pairsr[nd])


def mcflip(fls,Js,T,rs,Nltc,Nr,bc,ltype,fltype):
    for nf in range(Nr):
        r=random.choice(rs)
        flr=fls[ltc.rid(r,Nltc)]
        bc=1
        pairsr=ltc.pairs(r,Nltc[0],bc,ltype)
        e0=energyr(flr,fls,Js,pairsr,Nltc)
        flrf=flipsite(flr,fltype)
        e1=energyr(flrf,fls,Js,pairsr,Nltc)
        de=e1-e0
        if(de<0. or (de>0 and random.random()<math.e**(-de/T))):
            fls[ltc.rid(r,Nltc)]=flrf


def equilibrate(fls,Js,T,rs,Nltc,Nr,bc,ltype,fltype,NEQ):
    # Approach equilibrium
    [mcflip(fls,Js,T,rs,Nltc,Nr,bc,ltype,fltype) for n in range(NEQ)]
    

def sampling(fls,Js,T,rs,Nltc,Nr,bc,ltype,fltype,fm,NMC):
    '''
    Monte Carlo sampling
    '''
    mt=0.
    msqt=0.
    et=0.
    esqt=0.
    for n in range(NMC):
        mcflip(fls,Js,T,rs,Nltc,Nr,bc,ltype,fltype)
        mn=magnetization(fls,rs,Nltc,Nr,fm)
        mt+=mn
        msqt+=np.linalg.norm(mn)**2
        en=energy(fls,Js,rs,Nltc,Nr,ltype)
        et+=en
        esqt+=en**2
    chit=(msqt/NMC)-np.linalg.norm(mt/NMC)**2
    cTt=(esqt/NMC)-(et/NMC)**2
    return chit,cTt


def magnetization(fls,rs,Nltc,Nr,fm):
    return (1./Nr)*sum((fm**((r[0][0]-r[0][1])%2))*fls[ltc.rid(r,Nltc)] for r in rs)


def energy(fls,Js,rs,Nltc,Nr,ltype):
    e0=0.
    for r in rs:
        bc=1
        pairsr=ltc.pairs(r,Nltc[0],bc,ltype)
        er=energyr(fls[ltc.rid(r,Nltc)],fls,Js,pairsr,Nltc)
        er*=1./(2.*Nr)
        e0+=er
    return e0




