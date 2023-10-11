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
import brillouinzone as bz
sys.path.append('../tightbinding')
import tightbinding as tb
sys.path.append('../bandtheory')
import bandtheory as bdth




'''Plotting the bands'''


def mapfs(H,nf,ltype,prds,Nk,tosetde=False,de=0.):
    '''
    Obtain the Fermi surface for a given filling
    '''
    dataks=[]
    ks=bz.listbz(ltype,prds,Nk)[0]
    Hks=np.array([H(k) for k in ks])
    ees=np.linalg.eigvalsh(Hks)
    mu=bdth.fillingchempot(H,nf,ltype,prds,Nk)
    Nbd=np.shape(ees[0])[1]
    if(tosetde==False):de=1/Nk**2
    for n in range(Nbd):
        for nk in range(len(ks)):
            if(abs(ees[nk][n]-mu))<de:dataks.append([ks[nk][0],ks[nk][1],0.5])
    return dataks


def sectionband(H,mu,k1,k2,k0,Nk,toend=True):
    '''
    Compute the band energies from the Hamiltonian H along a momentum-space line section k1-k2.
    '''
    # Momenta from k1 to k2.
    k12s=list(np.linspace(k1,k2,num=Nk,endpoint=toend))
    # Momentum resolution.
    dk=np.linalg.norm(k12s[1]-k12s[0])
    # List the momenta to draw: k0 is the starting point of the [k1,k2] section in the plot along the high-symmetry-point contour. 
    kscs=np.array([k0+nk*dk for nk in range(len(k12s))])
    # Obtain the band energies at all momenta in the [k1,k2] section.
    Hks=np.array([H(k) for k in k12s])
    eescs=np.linalg.eigvalsh(Hks).T
    return kscs,eescs


def plotbandcontour(H,ltype,prds,Nfl,Nk,nf=0.,tosave=False,filetfig=''):
    '''
    Plot the band structure along a trajectory in the Brillouin zone.
    '''
    # Obtain the high-symmetry points.
    hsks=bz.hskcontour(ltype,prds)
    # Obtain the number of bands.
    Nbd=np.shape(H(hsks[0][1]))[0]
    # Determine the chemical potential mu that shows the filling. If no showing the filling, let nf=0 to plot all bands in blue.
    mu=bdth.fillingchempot(H,nf,ltype,prds,Nk)
    # Initial list of all plotted momenta.
    ks=np.array([])
    # Initial list of all plotted bands.
    bands=np.array([[] for n in range(Nbd)])
    # Initial point of plotted momentum.
    k0=0.
    # Initial list of high-symmetry points kts and their labels ktlbs along the plotted contour. These are the ticks of the x axis.
    kts,ktlbs=[k0],[hsks[0][0]]
    for nsc in range(len(hsks)-1):
        # Exclude the end point except in the last segment.
        toend=(nsc==len(hsks)-2)
        # Obtain the momenta and band energies in the section.
        kscs,eescs=sectionband(H,mu,hsks[nsc][1],hsks[nsc+1][1],k0,Nk,toend)
        # Add the momenta and band energies in the section to the overall lists.
        ks=np.concatenate((ks,kscs),axis=0)
        bands=np.concatenate((bands,eescs),axis=1)
        # Shift k0 to by the length of the [k1,k2] section.
        k0+=np.linalg.norm(hsks[nsc+1][1]-hsks[nsc][1])
        # Append the end momentum and its label to the list.
        kts+=[k0]
        ktlbs+=[hsks[nsc+1][0]]
    # Determine the colors of the data.
    def bandsegmentcolor(ee0,ee1,mu):
        # Determine the colors of a band segment [ee0,ee1].
        # Return green and blue below and above the chemical potential, respectively.
        if(ee0<mu+1e-14 and ee1<mu+1e-14):return 'g'
        else:return 'b'
    cs=[[bandsegmentcolor(bands[n,nk],bands[n,nk+1],mu) for nk in range(np.shape(bands)[1]-1)] for n in range(Nbd)]
    plt.rcParams.update({'font.size':30})
    for n in range(Nbd):
#        plt.scatter(ks,bands[n],s=2.,c=cs[n])
        # Obtain the collections of band segments.
        points=np.array([ks,bands[n]]).T.reshape(-1,1,2)
        segments=np.concatenate([points[:-1],points[1:]],axis=1)
        # Add the collection of band segments to the plot.
        lc=LineCollection(segments,colors=cs[n],linewidth=2)
        plt.gca().add_collection(lc)
        plt.gca().autoscale()
    # Set the ticks of the x axis as the high-symmetry points.
    [plt.axvline(x=hsk,color='k') for hsk in kts[1:-1]]
    plt.xlim(kts[0],kts[-1])
    plt.xticks(ticks=kts,labels=ktlbs)
    # Set the label of the y axis.
    plt.ylabel('$E_k$')
    plt.gcf()
    if(tosave==True):plt.savefig(filetfig,dpi=2000,bbox_inches='tight',pad_inches=0)
    plt.show()


def plotbz(ltype,prds,todata=False,dataks=[],tolabel=False,tosave=False,filetfig=''):
    '''
    Draw the Brillouin zone.
    '''
    # Type of Brillouin zone.
    bztype=bz.typeofbz(ltype,prds)
    # All high-symmetry points of the Brillouin zone. Specifying 2D.
    hsks=[[kp[0],np.array([kp[1][0],kp[1][1]])] for kp in bz.hskpoints(ltype,prds)]
    # Rectangular Brillouin zone.
    if(bztype=='rc'):
        # Corners of the Brillouin zone.
        bzcs=[hsks[3][1],hsks[4][1],-hsks[3][1],-hsks[4][1],]
        # High-symmetry points to label.
        hskls=[hsks[0],hsks[1],hsks[2],hsks[3]]
    # Hexagonal Brillouin zone.
    elif(bztype=='hx'):
        # Corners of the Brillouin zone.
        bzcs=[hsks[4][1],-hsks[6][1],hsks[5][1],-hsks[4][1],hsks[6][1],-hsks[5][1],]
        # High-symmetry points to label.
        hskls=[hsks[0],hsks[1],[hsks[5][0],-hsks[5][1]]]
    # Draw the edges of the Brillouin zone.
    plg=Polygon(bzcs,facecolor='none',edgecolor='k',linewidth=3)
    plt.rcParams.update({'font.size':30})
    fig,ax=plt.subplots()
    ax.add_patch(plg)
    # High-symmetry points to label.
    if(tolabel):
        hsklxs,hsklys=[hsk[1][0] for hsk in hskls],[hsk[1][1] for hsk in hskls]
        hskltxs,hskltys=[1.1*hsk[1][0] for hsk in hskls],[1.1*hsk[1][1] for hsk in hskls]
        hskltxs[0]+=0.1*hskls[-1][1][0]
        hskltys[0]+=0.1*hskls[-1][1][1]
        [plt.text(hskltxs[n],hskltys[n],hskls[n][0]) for n in range(len(hskls))]
        plt.scatter(hsklxs,hsklys,c='r')
    # If there is data to present, map it out.
    if(todata):
        [k0s,k1s,data]=np.array(dataks).transpose()
        plt.scatter(k0s,k1s,s=40.,c=data,cmap='coolwarm')
    ax=plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.axis('off')
    if(tosave==True):plt.savefig(filetfig,dpi=2000,bbox_inches='tight',pad_inches=0)
    plt.show()








