## Plot band module

'''Band structure module: Functions of plotting the bands'''

from math import *
import cmath as cmt
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size']=18
plt.rcParams.update({'figure.autolayout': True})
from matplotlib.patches import Polygon

import sys
sys.path.append('../lattice')
import lattice as ltc
sys.path.append('../tightbinding')
import tightbinding as tb
import bandtheory as bdth




'''Brillouin zone'''


def hskcontour(ltype,uctype):
    '''
    Set the high-symmetry points in the Brillouin zone forming the contour for the band structure.
    '''
    hska=ltc.hskpoints(ltype,uctype)
    if(ltype=='sq' and (uctype==111 or uctype==221)):
        return [hska[0],hska[1],hska[3],hska[2],hska[0]]
    if(uctype==211 or uctype==121):
        return [hska[0],hska[1],hska[3],hska[0],hska[2],hska[3],hska[0]]
    elif((ltype=='tr' or ltype=='ka') and (uctype==111 or uctype==221)):
        return [hska[0],hska[1],[hska[5][0],-hska[5][1]],hska[0]]
    elif((ltype=='tr' or ltype=='ka') and uctype==23231):
        return [hska[0],hska[1],[hska[5][0],hska[5][1]],hska[0]]


def brillouinzone(ltype,uctype,Nk):
    '''
    The momenta in the Brillouin zone.
    '''
    ks=[]
    if(ltype=='sq' or uctype==211 or uctype==121):
        kcs=[ltc.hskpoints(ltype,uctype)[n][1] for n in [1,2]]
        g0,g1=kcs[0],kcs[1]
        for n0 in np.linspace(-2.,2.,num=2*Nk):
            for n1 in np.linspace(-2.,2.,num=2*Nk):
                k=n0*g0+n1*g1
                if(-np.linalg.norm(kcs[0])**2-1e-14<=np.dot(k,kcs[0])<np.linalg.norm(kcs[0])**2+1e-14 and -np.linalg.norm(kcs[1])**2-1e-14<=np.dot(k,kcs[1])<np.linalg.norm(kcs[1])**2+1e-14):
                    ks.append(k)
    elif((ltype=='tr' or ltype=='ka') and (uctype==111 or uctype==221 or uctype==23231)):
        kcs=[ltc.hskpoints(ltype,uctype)[n][1] for n in [1,2,3]]
        g0,g1=kcs[0],kcs[1]
        for n0 in np.linspace(-2.,2.,num=2*Nk):
            for n1 in np.linspace(-2.,2.,num=2*Nk):
                k=n0*g0+n1*g1
                if(-np.linalg.norm(kcs[0])**2-1e-14<=np.dot(k,kcs[0])<np.linalg.norm(kcs[0])**2+1e-14 and -np.linalg.norm(kcs[1])**2-1e-14<=np.dot(k,kcs[1])<np.linalg.norm(kcs[1])**2+1e-14 and -np.linalg.norm(kcs[2])**2-1e-14<=np.dot(k,kcs[2])<np.linalg.norm(kcs[2])**2+1e-14):
                    ks.append(k)
    return ks


'''Plotting the bands'''


def fillingchempot(H,nf,ltype,uctype,Nbd,Nk):
    '''
    Compute the chemical potential for a given filling
    '''
    ks=brillouinzone(ltype,uctype,Nk)
    Hks=np.array([H(k) for k in ks])
    es=list(np.linalg.eigvalsh(Hks).flatten())
    es.sort()
    Nock=round(nf*len(es))
    mu=(es[Nock-1]+es[Nock])/2.
    print('mu = ',mu)
    return mu


def sectionband(H,mu,k1,k2,k0,Nk,toend,ks,bands):
    '''
    Compute the band eigenvalues along a momentum-space line section k1-k2.
    '''
    k12s=list(np.linspace(k1,k2,num=Nk,endpoint=toend))
    dk=np.linalg.norm(k12s[1]-k12s[0])
    for nk in range(len(k12s)):
        ks.append(k0+nk*dk)
        Hk=H(k12s[nk])
        ees=np.linalg.eigvalsh(Hk)
        [bands[n].append(ees[n]) for n in range(len(ees))]


def bandstructure(H,mu,ltype,uctype,Nfl,Nk,nf=0.):
    '''
    Plot the band structure along a trajectory in the Brillouin zone.
    '''
    Nbd=bdth.ucstnum(ltype,uctype,Nfl)
    if(nf>0.):
        mu=fillingchempot(H,nf,ltype,uctype,Nbd,Nk)
    hsks=hskcontour(ltype,uctype)
    ks=[]
    bands=[[] for n in range(Nbd)]
    k0=0.
    kts,ktlbs=[k0],[hsks[0][0]]
    for ns in range(len(hsks)-1):
        if(ns==len(hsks)-2):toend=True
        else:toend=False
        sectionband(H,mu,hsks[ns][1],hsks[ns+1][1],k0,Nk,toend,ks,bands)
        k0+=np.linalg.norm(hsks[ns+1][1]-hsks[ns][1])
        kts.append(k0)
        ktlbs.append(hsks[ns+1][0])
    cs=[['b' for nk in range(len(bands[n]))] for n in range(Nbd)]
    for n in range(Nbd):
        for nk in range(len(bands[n])):
            if(bands[n][nk]<mu+1e-14):
                cs[n][nk]='g'
    for n in range(Nbd):
        plt.scatter(ks,bands[n],s=2.,c=cs[n])
    [plt.axvline(x=hsk,color='k') for hsk in kts[1:-1]]
    plt.xlim(kts[0],kts[-1])
    plt.xticks(ticks=kts,labels=ktlbs)
    plt.ylabel('Ek')
    plt.gcf()
    plt.show()


def plotbz(ltype,uctype):
    '''
    Draw the Brillouin zone.
    '''
    hska=[np.array([kp[1][0],kp[1][1]]) for kp in ltc.hskpoints(ltype,uctype)]
    hskta=[kp[0] for kp in ltc.hskpoints(ltype,uctype)]
    if(ltype=='sq' or uctype==211 or uctype==121):
        bzcs=[hska[3],hska[4],-hska[3],-hska[4],]
        hskap=[hska[0],hska[1],hska[2],hska[3]]
        hsktap=[hskta[0],hskta[1],hskta[2],hskta[3]]
    elif((ltype=='tr' or ltype=='ka') and (uctype==111 or uctype==221)):
        bzcs=[hska[4],-hska[6],hska[5],-hska[4],hska[6],-hska[5],]
        hskap=[hska[0],hska[1],-hska[5]]
        hsktap=[hskta[0],hskta[1],hskta[5]]
    elif((ltype=='tr' or ltype=='ka') and uctype==23231):
        bzcs=[hska[4],-hska[6],hska[5],-hska[4],hska[6],-hska[5],]
        hskap=[hska[0],hska[1],hska[5]]
        hsktap=[hskta[0],hskta[1],hskta[5]]
    kmax=1.1*np.amax(abs(np.array(bzcs)))
    plg=Polygon(bzcs,facecolor='none',edgecolor='k',linewidth=3)
    fig,ax=plt.subplots()
    ax.add_patch(plg)
    hskapx,hskapy=[k[0] for k in hskap],[k[1] for k in hskap]
    hsktapx,hsktapy=[1.05*k[0] for k in hskap],[1.05*k[1] for k in hskap]
    hsktapx[0]+=0.05*hskap[-1][0]
    hsktapy[0]+=0.05*hskap[-1][1]
    for n in range(len(hskap)):
        plt.text(hsktapx[n],hsktapy[n],hsktap[n])
    plt.scatter(hskapx,hskapy,c='r')
    plt.ylim(-kmax,kmax)
    plt.xlim(-kmax,kmax)
    plt.xlabel('k$_x$')
    plt.ylabel('k$_y$')
#    ax.axis('off')
    plt.show()








