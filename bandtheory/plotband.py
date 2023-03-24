## Plot band module

'''Band structure module: Functions of plotting the bands'''

from math import *
import cmath as cmt
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size']=18
plt.rcParams.update({'figure.autolayout': True})
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
from mayavi import mlab

import sys
sys.path.append('../lattice')
import lattice as ltc
sys.path.append('../tightbinding')
import tightbinding as tb
import bandtheory as bdth




'''Plotting the bands'''


def fillingchempot(H,nf,ltype,uctype,Nk):
    '''
    Compute the chemical potential for a given filling
    '''
    ks=bdth.brillouinzone(ltype,uctype,Nk)[0]
    Hks=np.array([H(k) for k in ks])
    es=list(np.linalg.eigvalsh(Hks).flatten())
    es.sort()
    Nock=round(nf*len(es))
    mu=(es[Nock-1]+es[Nock])/2.
    print('mu = ',mu)
    return mu


def fermisurface(H,nf,ltype,uctype,Nk,tosetde=False,de=0.):
    '''
    Obtain the Fermi surface for a given filling
    '''
    dataks=[]
    ks=bdth.brillouinzone(ltype,uctype,Nk)[0]
    Hks=np.array([H(k) for k in ks])
    es=list(np.linalg.eigvalsh(Hks))
    mu=fillingchempot(H,nf,ltype,uctype,Nk)
    Nbd=np.shape(es[0])[0]
    if(tosetde==False):de=1/Nk**2
    for n in range(Nbd):
        for nk in range(len(ks)):
            if(abs(es[nk][n]-mu))<de:dataks.append([ks[nk][0],ks[nk][1],0.5])
    return dataks


def sectionband(H,mu,k1,k2,k0,Nk,toend,ks,bands):
    '''
    Compute the band energies along a momentum-space line section k1-k2.
    '''
    k12s=list(np.linspace(k1,k2,num=Nk,endpoint=toend))
    dk=np.linalg.norm(k12s[1]-k12s[0])
    for nk in range(len(k12s)):
        ks.append(k0+nk*dk)
        Hk=H(k12s[nk])
        ees=np.linalg.eigvalsh(Hk)
        [bands[n].append(ees[n]) for n in range(len(ees))]


def plotbandcontour(H,ltype,uctype,Nfl,Nk,nf=0.,tosave=False,filetfig=''):
    '''
    Plot the band structure along a trajectory in the Brillouin zone.
    '''
    Nbd=bdth.ucstnum(ltype,uctype,Nfl)
    mu=0.
    if(nf>0.):
        mu=fillingchempot(H,nf,ltype,uctype,Nk)
    hsks=bdth.hskcontour(ltype,uctype)
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
            if(bands[n][nk]<mu+1e-14 and bands[n][(nk+1)%len(bands[n])]<mu+1e-14):
                cs[n][nk]='g'
    plt.rcParams.update({'font.size':30})
    for n in range(Nbd):
        #plt.scatter(ks,bands[n],s=2.,c=cs[n])
        points=np.array([ks,bands[n]]).T.reshape(-1,1,2)
        segments=np.concatenate([points[:-1],points[1:]],axis=1)
        lc=LineCollection(segments,colors=cs[n],linewidth=2)
        plt.gca().add_collection(lc)
        lc.set_array(ks)
        plt.gca().add_collection(lc)
        plt.gca().autoscale()
    [plt.axvline(x=hsk,color='k') for hsk in kts[1:-1]]
    plt.xlim(kts[0],kts[-1])
    plt.xticks(ticks=kts,labels=ktlbs)
    plt.ylabel('$E_k$')
    plt.gcf()
    if(tosave==True):plt.savefig(filetfig,dpi=2000,bbox_inches='tight',pad_inches=0)
    plt.show()


def plotbz(ltype,uctype,todata=False,dataks=[],tosave=False,filetfig=''):
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
    plt.rcParams.update({'font.size':30})
    fig,ax=plt.subplots()
    ax.add_patch(plg)
    hskapx,hskapy=[k[0] for k in hskap],[k[1] for k in hskap]
    hsktapx,hsktapy=[1.1*k[0] for k in hskap],[1.1*k[1] for k in hskap]
    hsktapx[0]+=0.1*hskap[-1][0]
    hsktapy[0]+=0.1*hskap[-1][1]
    for n in range(len(hskap)):
        plt.text(hsktapx[n],hsktapy[n],hsktap[n])
    plt.scatter(hskapx,hskapy,c='r')
    if(todata):
        [ks0,ks1,datas]=np.array(dataks).transpose()
        plt.scatter(ks0,ks1,s=10.,c=datas,cmap='coolwarm')
    ax=plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.axis('off')
    #fig.axes.get_xaxis().set_visible(False)
    #fig.axes.get_yaxis().set_visible(False)
    #plt.ylim(-kmax,kmax)
    #plt.xlim(-kmax,kmax)
    #plt.xlabel('k$_x$')
    #plt.ylabel('k$_y$')
#    ax.axis('off')
    if(tosave==True):plt.savefig(filetfig,dpi=2000,bbox_inches='tight',pad_inches=0)
    plt.show()








