## Plot-lattice module

'''Plot-lattice module: Functions for plotting the lattice, as well as the charge and spin orders.'''

from math import *
import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab

import sys
sys.path.append('../lattice')
import lattice as ltc
sys.path.append('../tightbinding')
import tightbinding as tb




'''Plotting the lattice'''


def sites(rs,ltype,otype,os,res,tofull):
    '''
    Lattice site positions.
    '''
    ss=np.array([ltc.pos(r,ltype) for r in rs])
    [r0s,r1s,r2s]=ss.transpose()
    if(otype=='l'):os=np.array([0. for nr in range(len(rs))])
    elif(otype=='c'):os=np.array([(abs(ch)**0.75)*np.sign(ch) for ch in os])
    if(tofull):
        if(otype=='l' or otype=='c'):
            sspl=mlab.points3d(r0s,r1s,r2s,colormap='coolwarm',resolution=res,scale_factor=0.5)
            sspl.glyph.scale_mode='scale_by_vector'
            sspl.module_manager.scalar_lut_manager.use_default_range=False
            sspl.module_manager.scalar_lut_manager.data_range=[-1.,1.]
            sspl.mlab_source.dataset.point_data.scalars=os
        elif(otype=='s'):
            [s0,s1,s2]=np.array(os).transpose()
            sspl=mlab.quiver3d(r0s,r1s,r2s,s0,s1,s2,colormap='coolwarm',mode='arrow',scale_factor=1.,resolution=res/2)
            sspl.glyph.color_mode='color_by_scalar'
            sspl.module_manager.scalar_lut_manager.use_default_range=False
            sspl.module_manager.scalar_lut_manager.data_range=[-1.,1.]
            sspl.mlab_source.dataset.point_data.scalars=s2
    else:
        if(otype=='l' or otype=='c'):
            plt.scatter(r0s,r1s,s=1.,c=os,cmap='coolwarm',vmin=-1.,vmax=1.)
        if(otype=='s'):
            [s0,s1,s2]=np.array(os).transpose()
            plt.quiver(r0s,r1s,s0,s1,s2,cmap='coolwarm',angles='xy',scale_units='xy',scale=1)
            plt.clim(-1.,1.)


def bonds(rs,Nall,ltype,otype,os,res):
    '''
    Lattice bond positions.
    '''
    # Pairs at Bravais lattice site bls
    bc=0
    bs=[np.array([ltc.pos(pairt[0],ltype),ltc.pos(pairt[1],ltype)]).transpose() for r in rs for pairt in ltc.pairs(r,Nall[0][0],bc,ltype)[1]]
    if(otype=='l'):os=[np.array([0. for nb in range(len(bs))]),np.array([0. for nb in range(len(bs))])]
    elif(otype=='c'):os=[np.array([(abs(ch)**0.75)*np.sign(ch) for ch in os[0]]),np.array([(abs(ch)**0.5)*np.sign(ch) for ch in os[1]])]
    elif(otype=='s'):os=[np.array([(np.linalg.norm(sp)**0.75)*np.sign(sp[2]) for sp in os[0]]),np.array([(np.linalg.norm(sp)**0.5)*np.sign(sp[2]) for sp in os[1]])]
    # Imaginary
    bmp=[np.array([(b[n][0]+b[n][1])/2. for n in range(3)]) for b in bs]
    bv=[np.array([b[n][1]-b[n][0] for n in range(3)]) for b in bs]
    bii=np.array([bmp[nb]-(os[1][nb]/2.)*bv[nb] for nb in range(len(bs))]).transpose()
    biv=np.array([os[1][nb]*bv[nb] for nb in range(len(bs))]).transpose()
    bspli=mlab.quiver3d(bii[0],bii[1],bii[2],biv[0],biv[1],biv[2],color=(0.4660,0.6740,0.1880),mode='cone',scale_factor=1.,resolution=res/2)
    # Real
    bsplr=[mlab.plot3d(b[0],b[1],b[2],colormap='coolwarm',tube_radius=0.05,tube_sides=res/2) for b in bs]
    for n in range(len(bsplr)):
        bsplr[n].module_manager.scalar_lut_manager.use_default_range=False
        bsplr[n].module_manager.scalar_lut_manager.data_range=[-1.,1.]
        bsplr[n].mlab_source.dataset.point_data.scalars=[os[0][n],os[0][n]]


def plotlattice(rs,Nall,ltype,filetfig,otype='l',os=[[],[],[]],res=20,size=(5.,5.),setdpi=2000,tofull=True):
    '''
    Plot the lattice
    '''
    sos=os[0]
    bos=[os[1],os[2]]
    if(tofull):mlab.figure(bgcolor=None,size=(2000,2000))
    sites(rs,ltype,otype,sos,res,tofull)
    if(tofull):bonds(rs,Nall,ltype,otype,bos,res)
    if(tofull):
        mlab.view(azimuth=0.,elevation=0.)
        f=mlab.gcf()
        f.scene._lift()
        arr=mlab.screenshot(mode='rgba',antialiased=True)
        mlab.clf()
        mlab.close()
    '''
    fig=plt.figure()
    fig.set_size_inches(size)
    ax=plt.Axes(fig,[0.,0.,1.,1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(arr,interpolation='nearest')
    plt.savefig(filetfig,dpi=2000,bbox_inches='tight',pad_inches=0)
    plt.show()
    '''
    if(tofull):
        fig=plt.imshow(arr)
        plt.rcParams["figure.figsize"]=(size)
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
    else:
        plt.axis('off')
        ax=plt.gca()
        ax.set_aspect('equal', adjustable='box')
    plt.savefig(filetfig,dpi=setdpi,bbox_inches='tight',pad_inches=0,transparent=True)
    plt.clf()





def rescaledorder(chs,sps):
    '''
    Rescale the charge and spin orders for the plotting.
    '''
    omax=np.amax(np.array([chs[1],sps[1]]))
    return [np.array(chs[0][n])/omax for n in range(3)],[np.array(sps[0][n])/omax for n in range(3)]







