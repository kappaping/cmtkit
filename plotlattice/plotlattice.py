## Plot-lattice module

'''Plot-lattice module: Functions for plotting the lattice, as well as the charge and spin orders.'''

from math import *
import numpy as np
import matplotlib as mplb
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mayavi import mlab

import sys
sys.path.append('../lattice')
import lattice as ltc
sys.path.append('../tightbinding')
import tightbinding as tb
import densitymatrix as dm
import bogoliubovdegennes as bdg




'''Plotting the lattice'''


def sites(rs,rplids,ltype,odtype,odris,res,to3d,slcmapt=0):
    '''
    Lattice site positions.
    '''
    # Compute the positions of all lattice sites.
    ss=np.array([ltc.pos(rs[rplid],ltype) for rplid in rplids])
    [r0s,r1s,r2s]=ss.transpose()
    # For lattice plot: Set all of the order values = 0.
    cmapt='coolwarm'
    if(odtype=='l'):ods=np.array([0. for rplid in rplids])
    elif(odtype in ['sl','bislham']):
        ods=np.array([rs[rplid][1] for rplid in rplids])
        ods=ods-(ltc.slnum(ltype)-1)/2.
        ods=-0.7*ods/np.max(np.abs(ods))
        if(odtype=='sl'):cmapt='blue-red'
        elif(odtype=='bislham'):cmapt='PiYG'
    # For charge plot: Rescale the order magnitudes to enhance the plotting effect.
    elif(odtype=='c'):ods=np.array([(abs(od)**0.75)*np.sign(od) for od in odris[0]])
    # Plot
    if(to3d):
        if(odtype in ['l','sl','wf','bislham','c','fo']):
            sspl=mlab.points3d(r0s,r1s,r2s,colormap=cmapt,resolution=res,scale_factor=0.5)
            sspl.glyph.scale_mode='scale_by_vector'
            sspl.module_manager.scalar_lut_manager.use_default_range=False
            sspl.module_manager.scalar_lut_manager.data_range=[-1.,1.]
            sspl.mlab_source.dataset.point_data.scalars=ods
            if(odtype=='sl' and type(slcmapt)!=int):sspl.module_manager.scalar_lut_manager.lut.table=slcmapt
        elif(odtype=='s' or odtype=='fe'):
            [s0,s1,s2]=np.array(odris[0]).transpose()
            sspl=mlab.quiver3d(r0s,r1s,r2s,s0,s1,s2,colormap='coolwarm',mode='arrow',scale_factor=1.,resolution=res)
            sspl.glyph.color_mode='color_by_scalar'
            sspl.module_manager.scalar_lut_manager.use_default_range=False
            sspl.module_manager.scalar_lut_manager.data_range=[-1.,1.]
            sspl.mlab_source.dataset.point_data.scalars=s2
            sspl.glyph.glyph_source.glyph_source.shaft_radius=0.05
            sspl.glyph.glyph_source.glyph_source.tip_length=0.5
            sspl.glyph.glyph_source.glyph_source.tip_radius=0.1
    else:
        if(odtype=='l' or odtype=='c' or odtype=='fo'):
            plt.scatter(r0s,r1s,s=1.,c=odris[0],cmap='coolwarm',vmin=-1.,vmax=1.)
        if(odtype=='s'):
            [s0,s1,s2]=np.array(odris[0]).transpose()
            plt.quiver(r0s,r1s,s0,s1,s2,cmap='coolwarm',angles='xy',scale_units='xy',scale=1)
            plt.clim(-1.,1.)


def bonds(rs,nbplids,nnb,Nbl,ltype,bc,odtype,odris,res,slb):
    '''
    Lattice bond positions.
    '''
    # Set the non-periodic bond positions for the plotting
    bs=[]
    nptrs=ltc.periodictrsl(Nbl,bc)
    for pair in nbplids:
        r0,r1=rs[pair[0]],rs[pair[1]]
#        if(odtype=='wf' and slb!=None and r0[1]!=slb and r1[1]!=slb):toadd=False
#        else:toadd=True
        toadd=True
        if(toadd):
            r1dms=ltc.pairdist(ltype,r0,r1,True,nptrs)[1]
            bs+=[[ltc.pos(r0,ltype),ltc.pos(r1dm,ltype)] for r1dm in r1dms]
    blen=np.linalg.norm(bs[0][0]-bs[0][1])
    # For lattice plot: Set all of the order values = 0.
    cmapt='coolwarm'
    if(odtype in['l','sl''wf']):ods=[np.array([0. for nb in range(len(bs))]),np.array([0. for nb in range(len(bs))])]
    elif(odtype=='bislham'):
        if(nnb==1):
            ods=[np.array([-0.5 for nb in range(len(bs))]),np.array([0. for nb in range(len(bs))])]
            cmapt='black-white'
        elif(nnb==2):
            ods=[np.array([0.3*(-1)**rs[pair[0]][1] for pair in nbplids]),np.array([(abs(od)**0.4)*np.sign(od) for od in odris[1]])]
            cmapt='PiYG'
    # For charge plot: Rescale the order magnitudes to enhance the plotting effect.
    elif(odtype=='c'):ods=[np.array([(abs(od)**0.75)*np.sign(od) for od in odris[0]]),np.array([(abs(od)**0.4)*np.sign(od) for od in odris[1]])]
    elif(odtype=='s'):ods=[np.array([(np.linalg.norm(od)**0.75)*np.sign(od[2]) for od in odris[0]]),np.array([(np.linalg.norm(od)**0.4)*np.sign(od[2]) for od in odris[1]])]
    elif(odtype=='fo'):os=[np.array([(abs(fo)**0.75)*np.sign(fo) for fo in os[0] for n in range(2)]),np.array([(abs(fo)**0.4)*np.sign(fo) for fo in os[1] for n in range(2)])]
    elif(odtype=='fe'):os=[np.array([sgn*(np.linalg.norm(fe)**0.75)*np.sign(fe[2]) for fe in os[0] for sgn in [1,-1]]),np.array([sgn*(np.linalg.norm(fe)**0.4)*np.sign(fe[2]) for fe in os[1] for sgn in [1,-1]])]
    bmps=[(b[0]+b[1])/2. for b in bs]    # Middle points of bonds
    bvs=[b[0]-b[1] for b in bs]  # Bond vectors
    # Imaginary
    if(odtype in ['c','s']):
        biis=np.array([bmps[nb]-(ods[1][nb]/2.)*bvs[nb]/blen for nb in range(len(bs))]).transpose()  # Initial point of current cone
        bivs=np.array([ods[1][nb]*bvs[nb]/blen for nb in range(len(bs))]).transpose()   # Length of current cone
        bspli=mlab.quiver3d(biis[0],biis[1],biis[2],bivs[0],bivs[1],bivs[2],color=(0.4660,0.6740,0.1880),mode='cone',scale_factor=1.,resolution=res)
        bspli.glyph.glyph_source.glyph_source.angle=20
        bspli.glyph.glyph_source.glyph_source.height=0.6
    elif(odtype=='bislham'):
        biis=np.array([bmps[nb]-(ods[1][nb]/2.)*bvs[nb] for nb in range(len(bs))]).transpose()  # Initial point of current cone
        bivs=np.array([ods[1][nb]*bvs[nb] for nb in range(len(bs))]).transpose()   # Length of current cone
        bspli=mlab.quiver3d(biis[0],biis[1],biis[2],bivs[0],bivs[1],bivs[2],colormap=cmapt,mode='cone',scale_factor=1.,resolution=res)
        bspli.glyph.color_mode='color_by_scalar'
        bspli.module_manager.scalar_lut_manager.use_default_range=False
        bspli.module_manager.scalar_lut_manager.data_range=[-1.,1.]
        bspli.mlab_source.dataset.point_data.scalars=ods[0]
        bspli.glyph.glyph_source.glyph_source.angle=12
        bspli.glyph.glyph_source.glyph_source.height=0.8
    # Real
    if(odtype!='fo' and odtype!='fe'):
        [bvs0,bvs1,bvs2]=np.array(bvs).transpose()
        [bis0,bis1,bis2]=np.array([b[1] for b in bs]).transpose()
        rad=0.05/(blen*sqrt(nnb))
    if(odtype=='fo' or odtype=='fe'):
        bvs=[sgn*bv/2. for bv in bvs for sgn in [1,-1]]
        [bvs0,bvs1,bvs2]=np.array(bvs).transpose()
        [bis0,bis1,bis2]=np.array([bmp for bmp in bmps for n in range(2)]).transpose()
        rad=0.1
    bsplr=mlab.quiver3d(bis0,bis1,bis2,bvs0,bvs1,bvs2,colormap=cmapt,mode='cylinder',scale_factor=1.,resolution=res)
    bsplr.glyph.color_mode='color_by_scalar'
    bsplr.module_manager.scalar_lut_manager.use_default_range=False
    bsplr.module_manager.scalar_lut_manager.data_range=[-1.,1.]
    bsplr.mlab_source.dataset.point_data.scalars=ods[0]
    bsplr.glyph.glyph_source.glyph_source.radius=rad
#    bsplr=[mlab.plot3d(b[0],b[1],b[2],colormap='coolwarm',tube_radius=0.05,tube_sides=res/2) for b in bs]
#    for n in range(len(bsplr)):
#        bsplr[n].module_manager.scalar_lut_manager.use_default_range=False
#        bsplr[n].module_manager.scalar_lut_manager.data_range=[-1.,1.]
#        bsplr[n].mlab_source.dataset.point_data.scalars=[os[0][n],os[0][n]]


def plotlattice(rs,Nnb,nbplidss,Nbl,ltype,bc,filetfig,odtype='l',od2s=[[[],[]],[[],[]]],planes=[],arrows=[],texts=[],res=50,size=(5.,5.),dpi=300,to3d=True,show3d=False,plaz=0.,plel=0.,dist=None,slb=None,slcmapt=0,toscsh=True):
    '''
    Plot the lattice
    '''
    odriss=[[od2s[0][nnb],od2s[1][nnb]] for nnb in range(Nnb+1)]
    if(to3d):
        if(toscsh):mlab.figure(bgcolor=None,size=(2000,2000))
        else:mlab.figure(bgcolor=(1,1,1),size=(2000,2000))
    rplids=[nbplid[0] for nbplid in nbplidss[0]]
    sites(rs,rplids,ltype,odtype,odriss[0],res,to3d,slcmapt=slcmapt)
    if(to3d):
        for nnb in range(Nnb+1):
            if(nnb>0):bonds(rs,nbplidss[nnb],nnb,Nbl,ltype,bc,odtype,odriss[nnb],res,slb)
    if(to3d):
        if(len(planes)!=0):
            for plane in planes:
                # Define the corners of the parallelogram
                x,y,z=np.array(plane[0]).T
                cpl=plane[1]
                # Create the surface
                parallelogram=mlab.mesh(x.reshape((2, 2)),y.reshape((2, 2)),z.reshape((2, 2)),color=(cpl[0],cpl[1],cpl[2]),opacity=plane[2])
        if(len(arrows)!=0):
            for arrow in arrows:
                x,y,z=np.array(arrow[0]).T
                cars=arrow[1]
                curve=mlab.plot3d(x,y,z,color=(cars[0],cars[1],cars[2]),tube_radius=0.05,opacity=arrow[2],tube_sides=res)
                # Arrowhead at the end of the curve
                arrow_head_x=x[-1]
                arrow_head_y=y[-1]
                arrow_head_z=z[-1]
                # Arrowhead direction (tangent to the curve)
                dvec=np.array([x[-1]-x[-2],y[-1]-y[-2],z[-1]-z[-2]])
                dvec=dvec/np.linalg.norm(dvec)
                direction_x=dvec[0]
                direction_y=dvec[1]
                direction_z=dvec[2]
                # Plot the arrowhead using mlab.quiver3d
                arm=mlab.quiver3d(arrow_head_x,arrow_head_y,arrow_head_z,direction_x,direction_y,direction_z,mode='cone',scale_factor=1,color=(cars[0],cars[1],cars[2]),resolution=res,opacity=arrow[2])
                arm.glyph.glyph_source.glyph_source.angle=10.
                har=0.4
                arm.glyph.glyph_source.glyph_source.height=har
                arm.glyph.glyph_source.glyph_source.center=[har/2.,0.,0.]
    if(to3d):
        mlab.view(azimuth=plaz,elevation=plel,distance=dist,focalpoint=sum([ltc.pos(rs[rplid],ltype) for rplid in rplids])/len(rplids))
        f=mlab.gcf()
        f.scene._lift()
        if(toscsh):arr=mlab.screenshot(mode='rgba',antialiased=True)
        else:mlab.savefig('mayavi_screenshot.png')
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
    fig,ax=plt.subplots()
    if(to3d):
        if(toscsh):ax.imshow(arr)
        else:
            img=mpimg.imread('mayavi_screenshot.png')
            ax.imshow(img)
        plt.rcParams["figure.figsize"]=(size)
        ax.axis('off')
    else:
        plt.axis('off')
        ax=plt.gca()
        ax.set_aspect('equal', adjustable='box')
    if(len(texts)!=0):
        plt.rcParams.update({'font.size':15})
        for textt in texts:
            plt.text(textt[0][0],textt[0][1],textt[1],color=textt[2],horizontalalignment='center',verticalalignment='center',transform=ax.transAxes)
    plt.savefig(filetfig,dpi=dpi,bbox_inches='tight',pad_inches=0,transparent=True)
    plt.clf()




'''Plot the orders'''


def plotorder(P,ltype,rs,Nrfl,Nbl,bc,NB,rpls=[],Nnb=1,scl=1.,planes=[],arrows=[],texts=[],res=10,dpi=300,to3d=True,show3d=True,plaz=0.,plel=0.,dist=None,filetfig=[],tobdg=False,toscsh=True):
    '''
    Plot the orders.
    '''
    # Find out the n-th neighbor pairs.
    nbidss=[ltc.nthneighbors(nnb,NB) for nnb in range(Nnb+1)]
    # Compute the charge orders.
    od2s=dm.orders(P,nbidss,Nrfl,odtype='c',tobdg=tobdg)
    print('\n')
    print('Charge order:')
    def numth(nnb):
        if(nnb==1):return '-st'
        elif(nnb==2):return '-nd'
        elif(nnb==3):return '-rd'
        else:return '-th'
    for nnb in range(Nnb+1):
        print(nnb,numth(nnb),' neighbor:')
        print('R max = ',od2s[1][0][nnb],', R average = ',sum(od2s[0][0][nnb])/len(od2s[0][0][nnb]))
        print('I max = ',od2s[1][1][nnb],', I average = ',sum(od2s[0][1][nnb])/len(od2s[0][1][nnb]))
    od4s=[[od2s]]
    # If the flavor number > 1: Compute the spin orders.
    if(Nrfl[1]>1):
        od2s=dm.orders(P,nbidss,Nrfl,odtype='s',tobdg=tobdg)
        print('Spin order:')
        for nnb in range(Nnb+1):
            print(nnb,numth(nnb),' neighbor:')
            print('R max = ',od2s[1][0][nnb],', R average = ',sum(od2s[0][0][nnb])/len(od2s[0][0][nnb]))
            print('I max = ',od2s[1][1][nnb],', I average = ',sum(od2s[0][1][nnb])/len(od2s[0][1][nnb]))
        od4s[0]+=[od2s]
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
    od4sr=rescaledorder(od4s,scl)
    nbplidss,od4spl=collectplotelements(rs,nbidss,rpls,Nnb,od4sr)
    # Plot the charge orders.
    plotlattice(rs,Nnb,nbplidss,Nbl,ltype,bc,filetfig[0][0],'c',od4spl[0][0],planes=planes,arrows=arrows,texts=texts,res=res,dpi=dpi,to3d=to3d,show3d=show3d,plaz=plaz,plel=plel,dist=dist,toscsh=toscsh)
    # If the flavor number > 1: Plot the spin orders.
    if(Nrfl[1]>1):plotlattice(rs,Nnb,nbplidss,Nbl,ltype,bc,filetfig[0][1],'s',od4spl[0][1],res=res,dpi=dpi,to3d=to3d,show3d=show3d,plaz=plaz,plel=plel,dist=dist)
    # If the flavor number > 2: Plot the orbital orders.
    if(Nrfl[1]>2):plotlattice(rs,rplids,nbplids,Nbl,ltype,bc,filetfig[0][2],'s',odssspl[0][2],res=res,dpi=dpi,to3d=to3d,show3d=show3d,plaz=plaz,plel=plel,dist=dist)
    # Plot the pairing orders.
    if(tobdg):
        # Plot the flavor-even pairing orders.
        plotlattice(rs,rplids,nbplids,Nbl,ltype,bc,filetfig[1][0],'fe',odssspl[1][0],res=res,dpi=dpi,to3d=to3d,show3d=show3d,plaz=plaz,plel=plel,dist=dist)
        # If the flavor number > 2: Plot the flavor-odd pairing orders.
        if(Nrfl[1]>1):plotlattice(rs,rplids,nbplids,Nbl,ltype,bc,filetfig[1][1],'fo',odssspl[1][1],res=res,dpi=dpi,to3d=to3d,show3d=show3d,plaz=plaz,plel=plel,dist=dist)


def rescaledorder(od4s,scl):
    '''
    Rescale the orders for the plotting.
    '''
    odmax=max([np.max(np.array(od2s[1])) for od3s in od4s for od2s in od3s])
    return [[[[scl*np.array(ods)/odmax for ods in odris] for odris in od2s[0]] for od2s in od3s] for od3s in od4s]


def collectplotelements(rs,nbidss,rpls,Nnb,od4s):
    # Extract the orders to plot.
    if(len(rpls)==0):rpls=rs
    rplids=[ltc.siteid(rpl,rs) for rpl in rpls]
    od4spl=[[[[[] for ods in odris] for odris in od2s] for od2s in od3s] for od3s in od4s]
    nbplidss=[]
    rpl1ids=[rplid for rplid in rplids]
    for nnb in range(Nnb+1):
        nbids=nbidss[nnb]
        nbplids=[]
        for nb in range(len(nbids)):
            if(nbids[nb][0] in rplids):
                toaddnb=True
                if(nnb==1):
                    inrpl1ids=max([(nbids[nb][1]==rpl1id) for rpl1id in rpl1ids])
                    if(inrpl1ids==0):rpl1ids+=[nbids[nb][1]]
                elif(nnb>1):
                    inrpl1ids=np.prod(np.array([max([(nbids[nb][npn]==rpl1id) for rpl1id in rpl1ids]) for npn in range(2)]))
                    if(inrpl1ids==0):toaddnb=False
                if(toaddnb):
                    nbplids+=[nbids[nb]]
                    for nphpp in range(len(od4spl)):
                        for nodtype in range(len(od4spl[nphpp])):
                            for nri in range(len(od4spl[nphpp][nodtype])):
                                od4spl[nphpp][nodtype][nri][nnb]+=[od4s[nphpp][nodtype][nri][nnb][nb]]
        nbplidss+=[nbplids]
    return nbplidss,od4spl


def printcbar(filet,cmapt='coolwarm',cmapdarker=1.,cmapmax=1.,torevcmap=False):
    fig=plt.figure()
    ax=fig.add_axes([0.80,0.05,0.1,0.9])
    cmap=mplb.cm.get_cmap(cmapt)
    if(cmapdarker<1.):
        def darken_cmap(cmap, factor=cmapdarker):
            new_cmap=mplb.colors.LinearSegmentedColormap.from_list("dark_"+cmap.name,[(r*factor,g*factor,b*factor) for r,g,b,_ in cmap(np.linspace(0,1,256))])
            return new_cmap
        cmap=darken_cmap(cmap,factor=cmapdarker)
    if(cmapmax<1.):
        def newrange_cmap(cmap,cmapmax=cmapmax):
            new_cmap=mplb.colors.LinearSegmentedColormap.from_list("newmax_"+cmap.name,[cmap(0.5+cmapmax*(x-0.5)) for x in np.linspace(0,1,256)])
            return new_cmap
        cmap=newrange_cmap(cmap,cmapmax=cmapmax)
    if(torevcmap):
        def rev_cmap(cmap):
            new_cmap=mplb.colors.LinearSegmentedColormap.from_list("rev_"+cmap.name,[cmap(1-x) for x in np.linspace(0,1,256)])
            return new_cmap
        cmap=rev_cmap(cmap)
    cb=mplb.colorbar.ColorbarBase(ax,orientation='vertical',cmap=cmap)
    cb.set_ticks([])
    plt.savefig(filet,bbox_inches='tight')




