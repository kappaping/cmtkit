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
import densitymatrix as dm




'''Plotting the lattice'''


def sites(rs,ltype,otype,os,res,to3d):
    '''
    Lattice site positions.
    '''
    # Compute the positions of all lattice sites.
    ss=np.array([ltc.pos(r,ltype) for r in rs])
    [r0s,r1s,r2s]=ss.transpose()
    # For lattice plot: Set all of the order values = 0.
    if(otype=='l'):os=np.array([0. for nr in range(len(rs))])
    # For charge plot: Rescale the order magnitudes to enhance the plotting effect.
    elif(otype=='c'):os=np.array([(abs(ch)**0.75)*np.sign(ch) for ch in os])
    # Plot
    if(to3d):
        if(otype=='l' or otype=='c'):
            sspl=mlab.points3d(r0s,r1s,r2s,colormap='coolwarm',resolution=res,scale_factor=0.5)
            sspl.glyph.scale_mode='scale_by_vector'
            sspl.module_manager.scalar_lut_manager.use_default_range=False
            sspl.module_manager.scalar_lut_manager.data_range=[-1.,1.]
            sspl.mlab_source.dataset.point_data.scalars=os
        elif(otype=='s'):
            [s0,s1,s2]=np.array(os).transpose()
            sspl=mlab.quiver3d(r0s,r1s,r2s,s0,s1,s2,colormap='coolwarm',mode='arrow',scale_factor=1.,resolution=res)
            sspl.glyph.color_mode='color_by_scalar'
            sspl.module_manager.scalar_lut_manager.use_default_range=False
            sspl.module_manager.scalar_lut_manager.data_range=[-1.,1.]
            sspl.mlab_source.dataset.point_data.scalars=s2
            sspl.glyph.glyph_source.glyph_source.shaft_radius=0.05
            sspl.glyph.glyph_source.glyph_source.tip_length=0.5
            sspl.glyph.glyph_source.glyph_source.tip_radius=0.1
    else:
        if(otype=='l' or otype=='c'):
            plt.scatter(r0s,r1s,s=1.,c=os,cmap='coolwarm',vmin=-1.,vmax=1.)
        if(otype=='s'):
            [s0,s1,s2]=np.array(os).transpose()
            plt.quiver(r0s,r1s,s0,s1,s2,cmap='coolwarm',angles='xy',scale_units='xy',scale=1)
            plt.clim(-1.,1.)


def bonds(rs,nb1ids,Nbl,ltype,bc,otype,os,res):
    '''
    Lattice bond positions.
    '''
    # Set the non-periodic bond positions for the plotting
    rd1=min([np.linalg.norm(ltc.pos(rs[pair[0]],ltype)-ltc.pos(rs[pair[1]],ltype)) for pair in nb1ids])
    bs=[]
    nptrs=ltc.periodictrsl(Nbl,bc)
    for pair in nb1ids:
        r0,r1=rs[pair[0]],rs[pair[1]]
        r1dms=ltc.pairdist(ltype,r0,r1,True,nptrs)[1]
        bs+=[[ltc.pos(r0,ltype),ltc.pos(r1dm,ltype)] for r1dm in r1dms]
    # For lattice plot: Set all of the order values = 0.
    if(otype=='l'):os=[np.array([0. for nb in range(len(bs))]),np.array([0. for nb in range(len(bs))])]
    # For charge plot: Rescale the order magnitudes to enhance the plotting effect.
    elif(otype=='c'):os=[np.array([(abs(ch)**0.75)*np.sign(ch) for ch in os[0]]),np.array([(abs(ch)**0.4)*np.sign(ch) for ch in os[1]])]
    elif(otype=='s'):os=[np.array([(np.linalg.norm(sp)**0.75)*np.sign(sp[2]) for sp in os[0]]),np.array([(np.linalg.norm(sp)**0.4)*np.sign(sp[2]) for sp in os[1]])]
    # Imaginary
    bmp=[(b[0]+b[1])/2. for b in bs]    # Middle points of bonds
    bv=[b[1]-b[0] for b in bs]  # Bond vectors
    bii=np.array([bmp[nb]-(os[1][nb]/2.)*bv[nb] for nb in range(len(bs))]).transpose()  # Initial point of current cone
    biv=np.array([os[1][nb]*bv[nb] for nb in range(len(bs))]).transpose()   # Length of current cone
    bspli=mlab.quiver3d(bii[0],bii[1],bii[2],biv[0],biv[1],biv[2],color=(0.4660,0.6740,0.1880),mode='cone',scale_factor=1.,resolution=res)
    bspli.glyph.glyph_source.glyph_source.angle=20
    bspli.glyph.glyph_source.glyph_source.height=0.6
    # Real
    [bv0,bv1,bv2]=np.array(bv).transpose()
    [bi0,bi1,bi2]=np.array([b[0] for b in bs]).transpose()
    bsplr=mlab.quiver3d(bi0,bi1,bi2,bv0,bv1,bv2,colormap='coolwarm',mode='cylinder',scale_factor=1.,resolution=res)
    bsplr.glyph.color_mode='color_by_scalar'
    bsplr.module_manager.scalar_lut_manager.use_default_range=False
    bsplr.module_manager.scalar_lut_manager.data_range=[-1.,1.]
    bsplr.mlab_source.dataset.point_data.scalars=os[0]
    bsplr.glyph.glyph_source.glyph_source.radius=0.05
#    bsplr=[mlab.plot3d(b[0],b[1],b[2],colormap='coolwarm',tube_radius=0.05,tube_sides=res/2) for b in bs]
#    for n in range(len(bsplr)):
#        bsplr[n].module_manager.scalar_lut_manager.use_default_range=False
#        bsplr[n].module_manager.scalar_lut_manager.data_range=[-1.,1.]
#        bsplr[n].mlab_source.dataset.point_data.scalars=[os[0][n],os[0][n]]


def plotlattice(rs,nb1ids,Nbl,ltype,bc,filetfig,otype='l',os=[[],[],[]],res=50,size=(5.,5.),setdpi=2000,to3d=True,show3d=False,plaz=0.,plel=0.):
    '''
    Plot the lattice
    '''
    sos=os[0]
    bos=[os[1],os[2]]
    if(to3d):mlab.figure(bgcolor=None,size=(2000,2000))
    sites(rs,ltype,otype,sos,res,to3d)
    if(to3d):bonds(rs,nb1ids,Nbl,ltype,bc,otype,bos,res)
    if(to3d):
        mlab.view(azimuth=plaz,elevation=plel)
        f=mlab.gcf()
        f.scene._lift()
        arr=mlab.screenshot(mode='rgba',antialiased=True)
        if(show3d):mlab.show()
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
    if(to3d):
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







