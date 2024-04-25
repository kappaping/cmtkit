## Plot-lattice module

'''Plot-lattice module: Functions for plotting the lattice, as well as the charge and spin orders.'''

from math import *
import numpy as np
import matplotlib as mplb
import matplotlib.pyplot as plt
from mayavi import mlab

import sys
sys.path.append('../lattice')
import lattice as ltc
sys.path.append('../tightbinding')
import tightbinding as tb
import densitymatrix as dm
import bogoliubovdegennes as bdg




'''Plotting the lattice'''


def sites(rs,rplids,ltype,otype,os,res,to3d):
    '''
    Lattice site positions.
    '''
    # Compute the positions of all lattice sites.
    ss=np.array([ltc.pos(rs[rplid],ltype) for rplid in rplids])
    [r0s,r1s,r2s]=ss.transpose()
    # For lattice plot: Set all of the order values = 0.
    if(otype=='l'):os=np.array([0. for rplid in rplids])
    # For charge plot: Rescale the order magnitudes to enhance the plotting effect.
    elif(otype=='c'):os=np.array([(abs(ch)**0.75)*np.sign(ch) for ch in os])
    # Plot
    if(to3d):
        if(otype=='l' or otype=='c' or otype=='fo'):
            sspl=mlab.points3d(r0s,r1s,r2s,colormap='coolwarm',resolution=res,scale_factor=0.5)
            sspl.glyph.scale_mode='scale_by_vector'
            sspl.module_manager.scalar_lut_manager.use_default_range=False
            sspl.module_manager.scalar_lut_manager.data_range=[-1.,1.]
            sspl.mlab_source.dataset.point_data.scalars=os
        elif(otype=='s' or otype=='fe'):
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
        if(otype=='l' or otype=='c' or otype=='fo'):
            plt.scatter(r0s,r1s,s=1.,c=os,cmap='coolwarm',vmin=-1.,vmax=1.)
        if(otype=='s'):
            [s0,s1,s2]=np.array(os).transpose()
            plt.quiver(r0s,r1s,s0,s1,s2,cmap='coolwarm',angles='xy',scale_units='xy',scale=1)
            plt.clim(-1.,1.)


def bonds(rs,nbplids,Nbl,ltype,bc,otype,os,res):
    '''
    Lattice bond positions.
    '''
    # Set the non-periodic bond positions for the plotting
    bs=[]
    nptrs=ltc.periodictrsl(Nbl,bc)
    for pair in nbplids:
        r0,r1=rs[pair[0]],rs[pair[1]]
        r1dms=ltc.pairdist(ltype,r0,r1,True,nptrs)[1]
        bs+=[[ltc.pos(r0,ltype),ltc.pos(r1dm,ltype)] for r1dm in r1dms]
    # For lattice plot: Set all of the order values = 0.
    if(otype=='l'):os=[np.array([0. for nb in range(len(bs))]),np.array([0. for nb in range(len(bs))])]
    # For charge plot: Rescale the order magnitudes to enhance the plotting effect.
    elif(otype=='c'):os=[np.array([(abs(ch)**0.75)*np.sign(ch) for ch in os[0]]),np.array([(abs(ch)**0.4)*np.sign(ch) for ch in os[1]])]
    elif(otype=='s'):os=[np.array([(np.linalg.norm(sp)**0.75)*np.sign(sp[2]) for sp in os[0]]),np.array([(np.linalg.norm(sp)**0.4)*np.sign(sp[2]) for sp in os[1]])]
    elif(otype=='fo'):os=[np.array([(abs(fo)**0.75)*np.sign(fo) for fo in os[0] for n in range(2)]),np.array([(abs(fo)**0.4)*np.sign(fo) for fo in os[1] for n in range(2)])]
    elif(otype=='fe'):os=[np.array([sgn*(np.linalg.norm(fe)**0.75)*np.sign(fe[2]) for fe in os[0] for sgn in [1,-1]]),np.array([sgn*(np.linalg.norm(fe)**0.4)*np.sign(fe[2]) for fe in os[1] for sgn in [1,-1]])]
    bmps=[(b[0]+b[1])/2. for b in bs]    # Middle points of bonds
    bvs=[b[0]-b[1] for b in bs]  # Bond vectors
    # Imaginary
    if(otype=='c' or otype=='s'):
        biis=np.array([bmps[nb]-(os[1][nb]/2.)*bvs[nb] for nb in range(len(bs))]).transpose()  # Initial point of current cone
        bivs=np.array([os[1][nb]*bvs[nb] for nb in range(len(bs))]).transpose()   # Length of current cone
        bspli=mlab.quiver3d(biis[0],biis[1],biis[2],bivs[0],bivs[1],bivs[2],color=(0.4660,0.6740,0.1880),mode='cone',scale_factor=1.,resolution=res)
        bspli.glyph.glyph_source.glyph_source.angle=20
        bspli.glyph.glyph_source.glyph_source.height=0.6
    # Real
    if(otype!='fo' and otype!='fe'):
        [bvs0,bvs1,bvs2]=np.array(bvs).transpose()
        [bis0,bis1,bis2]=np.array([b[1] for b in bs]).transpose()
        rad=0.05
    if(otype=='fo' or otype=='fe'):
        bvs=[sgn*bv/2. for bv in bvs for sgn in [1,-1]]
        [bvs0,bvs1,bvs2]=np.array(bvs).transpose()
        [bis0,bis1,bis2]=np.array([bmp for bmp in bmps for n in range(2)]).transpose()
        rad=0.1
    bsplr=mlab.quiver3d(bis0,bis1,bis2,bvs0,bvs1,bvs2,colormap='coolwarm',mode='cylinder',scale_factor=1.,resolution=res)
    bsplr.glyph.color_mode='color_by_scalar'
    bsplr.module_manager.scalar_lut_manager.use_default_range=False
    bsplr.module_manager.scalar_lut_manager.data_range=[-1.,1.]
    bsplr.mlab_source.dataset.point_data.scalars=os[0]
    bsplr.glyph.glyph_source.glyph_source.radius=rad
#    bsplr=[mlab.plot3d(b[0],b[1],b[2],colormap='coolwarm',tube_radius=0.05,tube_sides=res/2) for b in bs]
#    for n in range(len(bsplr)):
#        bsplr[n].module_manager.scalar_lut_manager.use_default_range=False
#        bsplr[n].module_manager.scalar_lut_manager.data_range=[-1.,1.]
#        bsplr[n].mlab_source.dataset.point_data.scalars=[os[0][n],os[0][n]]


def plotlattice(rs,rplids,nbplids,Nbl,ltype,bc,filetfig,otype='l',os=[[],[],[]],res=50,size=(5.,5.),dpi=300,to3d=True,show3d=False,plaz=0.,plel=0.):
    '''
    Plot the lattice
    '''
    sos=os[0]
    bos=[os[1],os[2]]
    if(to3d):mlab.figure(bgcolor=None,size=(2000,2000))
    sites(rs,rplids,ltype,otype,sos,res,to3d)
    if(to3d):bonds(rs,nbplids,Nbl,ltype,bc,otype,bos,res)
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
    plt.savefig(filetfig,dpi=dpi,bbox_inches='tight',pad_inches=0,transparent=True)
    plt.clf()




'''Plot the orders'''


def plotorder(P,ltype,rs,Nrfl,Nbl,bc,NB,rpls=[],scl=1.,res=10,dpi=300,to3d=True,show3d=True,plaz=0.,plel=0.,filetfig=[],tobdg=False):
    '''
    Plot the orders.
    '''
    # Find out the first-neighbor pairs.
    nb1ids=ltc.nthneighbors(1,NB)
    # Compute the charge orders.
    chs=dm.chargeorder(P,nb1ids,Nrfl,tobdg)
    print('charge order')
    print('site order max = ',chs[1][0],', site order average = ',sum(chs[0][0])/len(chs[0][0]))
    print('real bond order max = ',chs[1][1],', real bond order average = ',sum(chs[0][1])/len(chs[0][1]))
    print('imaginary bond order max = ',chs[1][2],', imaginary bond order average = ',sum(chs[0][2])/len(chs[0][2]))
    odsss=[[chs]]
    # If the flavor number > 1: Compute the spin orders.
    if(Nrfl[1]>1):
        sps=dm.spinorder(P,nb1ids,Nrfl,tobdg)
        print('spin order')
        print('site order max = ',sps[1][0],', site order average = ',sum(sps[0][0])/len(sps[0][0]))
        print('real bond order max = ',sps[1][1],', real bond order average = ',sum(sps[0][1])/len(sps[0][1]))
        print('imaginary bond order max = ',sps[1][2],', imaginary bond order average = ',sum(sps[0][2])/len(sps[0][2]))
        odsss[0]+=[sps]
    # If the flavor number > 2: Compute the orbital orders.
    if(Nrfl[1]>2):
        ors=dm.orbitalorder(P,nb1ids,Nrfl,tobdg)
        print('orbital order')
        print('site order max = ',ors[1][0],', site order average = ',sum(ors[0][0])/len(ors[0][0]))
        print('real bond order max = ',ors[1][1],', real bond order average = ',sum(ors[0][1])/len(ors[0][1]))
        print('imaginary bond order max = ',ors[1][2],', imaginary bond order average = ',sum(ors[0][2])/len(ors[0][2]))
        odsss[0]+=[ors]
    # Compute the pairing orders.
    if(tobdg):
        # Compute the flavor-even orders.
        feps=bdg.flavorevenpairingorder(P,nb1ids,Nrfl)
        print('Flavor-even pairing order')
        print('site order max = ',feps[1][0],', site order average = ',sum(feps[0][0])/len(feps[0][0]))
        print('real bond order max = ',feps[1][1],', real bond order average = ',sum(feps[0][1])/len(feps[0][1]))
        print('imaginary bond order max = ',feps[1][2],', imaginary bond order average = ',sum(feps[0][2])/len(feps[0][2]))
        odsss+=[[feps]]
        # If the flavor number > 1: Compute the flavor-odd orders.
        if(Nrfl[1]>1):
            fops=bdg.flavoroddpairingorder(P,nb1ids,Nrfl)
            print('Flavor-odd pairing order')
            print('site order max = ',fops[1][0],', site order average = ',sum(fops[0][0])/len(fops[0][0]))
            print('real bond order max = ',fops[1][1],', real bond order average = ',sum(fops[0][1])/len(fops[0][1]))
            print('imaginary bond order max = ',fops[1][2],', imaginary bond order average = ',sum(fops[0][2])/len(fops[0][2]))
            odsss[1]+=[fops]
    # Rescale the orders.
    odsssr=rescaledorder(odsss,scl)
    # Extract the orders to plot.
    if(len(rpls)==0):rpls=rs
    rplids=[ltc.siteid(rpl,rs) for rpl in rpls]
    odssspl=[[[[],[],[]] for os in oss] for oss in odsss]
    for m0 in range(len(odssspl)):
        for m1 in range(len(odssspl[m0])):
            odssspl[m0][m1][0]=[odsssr[m0][m1][0][rplid] for rplid in rplids]
    nbplids=[]
    for nb in range(len(nb1ids)):
        if(nb1ids[nb][0] in rplids):
            nbplids+=[nb1ids[nb]]
            for m0 in range(len(odssspl)):
                for m1 in range(len(odssspl[m0])):
                    odssspl[m0][m1][1]+=[odsssr[m0][m1][1][nb]]
                    odssspl[m0][m1][2]+=[odsssr[m0][m1][2][nb]]
    # Plot the charge orders.
    plotlattice(rs,rplids,nbplids,Nbl,ltype,bc,filetfig[0][0],'c',odssspl[0][0],res=res,dpi=dpi,to3d=to3d,show3d=show3d,plaz=plaz,plel=plel)
    # If the flavor number > 1: Plot the spin orders.
    if(Nrfl[1]>1):plotlattice(rs,rplids,nbplids,Nbl,ltype,bc,filetfig[0][1],'s',odssspl[0][1],res=res,dpi=dpi,to3d=to3d,show3d=show3d,plaz=plaz,plel=plel)
    # If the flavor number > 2: Plot the orbital orders.
    if(Nrfl[1]>2):plotlattice(rs,rplids,nbplids,Nbl,ltype,bc,filetfig[0][2],'s',odssspl[0][2],res=res,dpi=dpi,to3d=to3d,show3d=show3d,plaz=plaz,plel=plel)
    # Plot the pairing orders.
    if(tobdg):
        # Plot the flavor-even pairing orders.
        plotlattice(rs,rplids,nbplids,Nbl,ltype,bc,filetfig[1][0],'fe',odssspl[1][0],res=res,dpi=dpi,to3d=to3d,show3d=show3d,plaz=plaz,plel=plel)
        # If the flavor number > 2: Plot the flavor-odd pairing orders.
        if(Nrfl[1]>1):plotlattice(rs,rplids,nbplids,Nbl,ltype,bc,filetfig[1][1],'fo',odssspl[1][1],res=res,dpi=dpi,to3d=to3d,show3d=show3d,plaz=plaz,plel=plel)


def rescaledorder(osss,scl):
    '''
    Rescale the orders for the plotting.
    '''
    omax=max([np.max(np.array([os[1] for os in oss])) for oss in osss])
    return [[[scl*np.array(os[0][n])/omax for n in range(3)] for os in oss] for oss in osss]


def printcbar(filet,cmap='coolwarm'):
    fig=plt.figure()
    ax=fig.add_axes([0.80,0.05,0.1,0.9])
    cb=mplb.colorbar.ColorbarBase(ax,orientation='vertical',cmap=cmap)
    cb.set_ticks([])
    plt.savefig(filet,bbox_inches='tight')




