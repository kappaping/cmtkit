## Time-dependent Hartree-Fock module

'''Time-dependent Hartree-Fock module: Functions of time-dependent Hartree-Fock theory'''

from math import *
import cmath as cmt
import numpy as np
import matplotlib as mplb
import matplotlib.pyplot as plt
plt.rcParams['font.size']=18
plt.rcParams.update({'figure.autolayout': True})
import joblib
from scipy.interpolate import make_interp_spline
from scipy.stats import linregress
from scipy.optimize import curve_fit

import sys
sys.path.append('../../cmt_code/lattice')
import lattice as ltc
sys.path.append('../../cmt_code/tightbinding')
import tightbinding as tb
import densitymatrix as dm
sys.path.append('../../cmt_code/bandtheory')
import bandtheory as bdth
import hartreefock as hf




'''Time-varying functions.'''


def peierls(tt,H0,NB,RDV,Nrfl,Af=np.array([0.,0.,0.]),toef=False,F0=1e-4,Dttef=25.,taut=5.,toprinttvar=False):
    if(callable(Af)):Aft=Af(tt,toprinttvar)
    if(toef):
        if(toprinttvar):print('Test E field: F0 =',F0,', Dttef =',Dttef,', taut =',taut)
        if(tt>=(ttc+Dttef)):Aft=Aft+F0*((tt-(ttc+Dttef))+taut*(e**(-(tt-(ttc+Dttef))/taut))-taut)*np.array([1.,0.,0.])
    APl=np.array([[e**(-1.j*np.dot(Aft,RDV[rid0,rid1])) for rid1 in range(Nrfl[0]) for fl1 in range(Nrfl[1])] for rid0 in range(Nrfl[0]) for fl0 in range(Nrfl[1])])
    return [H0*APl,Aft]


def pumppulse(tt,poltype,epol,omegat,Ac,ttc,sigmat,toprinttvar=False):
    if(toprinttvar):
        print('Pump pulse: poltype =',poltype,end='')
        if(poltype=='lin'):print(', epol =',epol,end='')
        print(', omegat =',omegat,', Ac =',Ac,', ttc =',ttc,', sigmat =',sigmat)
    exptt=((tt-ttc)**2)/(2.*(sigmat**2))
    if(exptt>60.):Aft=np.array([0.,0.,0.])
    else:
        if(poltype=='lin'):Aft=Ac*(e**(-exptt))*cos(omegat*(tt-ttc))*epol
        elif(poltype=='cir'):Aft=Ac*(e**(-exptt))*np.array([cos(omegat*(tt-ttc)),sin(omegat*(tt-ttc)),0.])
    return Aft




'''Algorithm'''




def timedephartreefock(P0,H0,UINT,rs,NB,RDV,Nrfl,Nst,nf,timevar,filet,dtt=1e-2,ttmax=1000,writett=0.1,writedenmattt=20.,totesttt=False,toreadevol=False,filetevol='',tosavedm=False,toorders=False,orders=0,toorddev=False,tochn=False,chernnum=0,tohall=False,timevaref=0.):
    '''
    Algorithm of Hartree-Fock approximation.
    sbmaxs: [schsmaxs,bchsrmaxs,bchsimaxs,sspsmaxs,bspsrmaxs,bspsimaxs]
    '''

    '''
    Initialize.
    '''
    # Initial density matrix.
    tt=0.
    P0=P0
    Pm=P0
    Hhfm=hf.hfham(H0,Pm,UINT)
    if(tohall):Pefm,Hhfefm=Pm,Hhfm
    tts,Afts=[],[]
    eems,Pms=[],[]
    ordrsss,ordisss=[],[]
    nb1idss,FFi=neighbors(NB,RDV)
    sgdordrs,sgdordis=[],[]
    chns=[]
    Yin=ybdmat(RDV,Nrfl)
    sigmaxys=[]
    if(toreadevol):
        print('Read the time evolution from:',filetevol)
        [Pm,Pms,tts,Afts,eems,ordrsss,ordisss,sgdordrs,sgdordis,chns,sigmaxys]=joblib.load(filetevol)
        tt=tts[-1]
    else:print('Initialize a new time evolution.')
    print('Start iteration.')
    sys.stdout.flush()
    '''
    Start the iterative variation.
    '''
    NT=round(ttmax/dtt)
    for nm in range(NT+1):
        if(tt>ttmax+1e-8):break
        if((abs((tt/writett)-round(tt/writett))<1e-8) and ((toreadevol and (nm==0))==False)):towrite=True
        else:towrite=False
        # Save the density matrix.
        if(tosavedm and (abs((tt/writedenmattt)-round(tt/writedenmattt))<1e-8) and ((toreadevol and (nm==0))==False)):
            print('Add density matrix to save list.')
            Pms+=[[tt,Pm]]
        else:Pms=[]
        # Setup the Hartree-Fock Hamiltonian.
        [H0t,Aft,UINTt]=timevar(tt,H0,UINT,toprinttvar=towrite)
        Hhfm=hf.hfham(H0t,Pm,UINTt)
        if(tohall):
            [H0eft,Afeft,UINTeft]=timevaref(tt,H0,UINT,toprinttvar=towrite)
            Hhfefm=hf.hfham(H0eft,Pefm,UINTeft)
        if(towrite):
            print('Write to:',filet)
            tts+=[tt]
            Afts+=[Aft]
            eem=hf.energy(H0t,Pm,Hhfm,Nst)
            eems+=[eem]
            if(toorders):
                ordrfssm,ordifssm=orders(Pm)
                ordrsss+=[ordrfssm]
                ordisss+=[ordifssm]
                if(toorddev):
                    avgdordrm,avgdordim,sgdordrm,sgdordim=orderdev(Pm,rs,NB,nb1idss,FFi)[0:4]
                    sgdordrs+=[sgdordrm]
                    sgdordis+=[sgdordim]
            if(tochn):
                chnm=chernnum(Hhfm)
                chns+=[chnm]
            if(tohall):
                sigmaxym=(2.*pi)*(ycurrent(Pefm,H0eft,Yin,Nst)-ycurrent(Pm,H0t,Yin,Nst))/(1e-4)
                sigmaxys+=[sigmaxym]
            joblib.dump([Pm,Pms,tts,Afts,eems,ordrsss,ordisss,sgdordrs,sgdordis,chns,sigmaxys],filet)
        # Discrete time evolution.
        Pm,dppnp=timeevolrk4(Pm,tt,H0,UINT,timevar,Hhfm,dtt,Nst,nf)
        if(tohall):Pefm,dppnpef=timeevolrk4(Pm,tt,H0,UINT,timevaref,Hhfefm,dtt,Nst,nf)
        # Print out the status.
        print(nm,'-th iteration, t = ',round(tt,8),', |Pm^2-Pm| = ',dppnp,end='')
        if(totesttt):print(', |Pm-P0| =',np.max(np.abs(Pm-P0)),end='')
        if(tohall):print(', |Pefm^2-Pefm| = ',dppnpef,end='')
        if(towrite):
            print('\n    eem =',eem,end='')
            if(toorders):
                print('\n    ordrfssm =',ordrfssm,', ordifssm =',ordifssm,end='')
                if(toorddev):
                    print('\n    avgdordrm =',avgdordrm,', sgdordrm =',sgdordrm,', avgdordim',avgdordim,', sgdordim =',sgdordim,end='')
                if(tochn):print('\n    chnm =',chnm,',',end='')
                if(tohall):print('\n    sigmaxym =',sigmaxym,end='')
        print('')
        sys.stdout.flush()
        tt+=dtt
        tt=round(tt/dtt)*dtt
    # Write out the spectrum evolution.
    joblib.dump([Pm,Pms,tts,Afts,eems,ordrsss,ordisss,sgdordrs,sgdordis,chns,sigmaxys],filet)
    # Print the final status
    print('End: ',nm,'-th iteration')
    sys.stdout.flush()


def timeevolrk4(Pm,tt,H0,UINT,timevar,Hhfm,dtt,Nst,nf):
    '''
    Perform the discrete time evolution with the Runge-Kutta 4-th-order method (RK4).
    For a differential equation dy/dt=f(t,y), the RK4 method is as follows:
    yn=y(tn)
    k1=f(tn,yn)
    k2=f(tn+dt/2,yn+(dt/2)k1)
    k3=f(tn+dt/2,yn+(dt/2)k2)
    k4=f(tn+dt,yn+(dt)k3)
    y(n+1)=y(t(n+1))=yn+(dt/6)(k1+2k2+2k3+k4)
    Our time-evolution equation: dP/dt=-i[Hhf,P]
    '''
    # define the time-evolution equation.
    def ddP(P,Hhf):return -1.j*(np.dot(Hhf,P)-np.dot(P,Hhf))
    # Compute k1.
    ddPmt1=ddP(Pm,Hhfm)
    # Compute k2.
    [H0t12,Aft12,UINTt12]=timevar(tt+dtt/2.,H0,UINT,toprinttvar=False)
    Pmt1=Pm+(dtt/2.)*ddPmt1
    Hhfmt1=hf.hfham(H0t12,Pmt1,UINTt12)
    ddPmt2=ddP(Pmt1,Hhfmt1)
    # Compute k3.
    Pmt2=Pm+(dtt/2.)*ddPmt2
    Hhfmt2=hf.hfham(H0t12,Pmt2,UINTt12)
    ddPmt3=ddP(Pmt2,Hhfmt2)
    # Compute k4.
    [H0t1,Aft1,UINTt1]=timevar(tt+dtt,H0,UINT,toprinttvar=False)
    Pmt3=Pm+dtt*ddPmt3
    Hhfmt3=hf.hfham(H0t1,Pmt3,UINTt1)
    ddPmt4=ddP(Pmt3,Hhfmt3)
    # Compute Pm.
    Pm=Pm+(dtt/6.)*(ddPmt1+2.*ddPmt2+2.*ddPmt3+ddPmt4)
    # Correct the projector condition P^2=P.
    dppnp=np.max(np.abs(np.dot(Pm,Pm)-Pm))
    if(dppnp>5e-15):
        Up=np.linalg.eigh(Pm)[1]
        Noc=round(Nst*nf)
        D=np.diag(np.array((Nst-Noc)*[0.]+Noc*[1.]))
        Pmt=np.linalg.multi_dot([Up,D,Up.conj().T])
        print('Correct the projector condition: |Pmt-Pm| =',np.max(np.abs(Pmt-Pm)))
        Pm=Pmt
        dppnp=np.max(np.abs(np.dot(Pm,Pm)-Pm))
    return Pm,dppnp


def timeevolunitary(Pm,Hhfm,dtt,Nst,nf):
    # Unitary discrete time evolution.
    ees,Uee=np.linalg.eigh(Hhfm)
    Ut=np.linalg.multi_dot([Uee,np.diag([e**(-1.j*dtt*ee) for ee in ees]),Uee.conj().T])
    Pm=np.linalg.multi_dot([Ut,Pm,Ut.conj().T])
    # Correct the projector condition P^2=P.
    dppnp=np.max(np.abs(np.dot(Pm,Pm)-Pm))
    if(dppnp>5e-15):
        Up=np.linalg.eigh(Pm)[1]
        Noc=round(Nst*nf)
        D=np.diag(np.array((Nst-Noc)*[0.]+Noc*[1.]))
        Pmt=np.linalg.multi_dot([Up,D,Up.conj().T])
        print('Correct the projector condition: |Pmt-Pm| =',np.max(np.abs(Pmt-Pm)))
        Pm=Pmt
        dppnp=np.max(np.abs(np.dot(Pm,Pm)-Pm))
    return Pm,dppnp


def orderparameter(Pm,kffrss,kffiss,ltype,rs,r0c,nbidss,RDV,Nst):
    ordrfss=[]
    Nsl=ltc.slnum(ltype)
    for kffrs in kffrss:
        ordrts=[sum([sum([kffrs[2][nkr][rs[nbid[0]][1],rs[nbid[1]][1]]*Pm[nbid[0],nbid[1]]*(Nsl/Nst)*e**(1.j*(np.dot(ltc.pos(rs[nbid[0]],ltype)-RDV[nbid[0],nbid[1]]-r0c,kffrs[1][nkr]+kffrs[3][nkr]*kffrs[0])-np.dot(ltc.pos(rs[nbid[0]],ltype)-r0c,kffrs[1][nkr]))) for nbid in nbids]) for nbids in nbidss]) for nkr in range(len(kffrs[1]))]
        ordrfs=[np.dot(np.array(ordrts),np.array(ff)).real for ff in kffrs[4]]
        ordrfss+=[ordrfs]
    ordifss=[]
    for kffis in kffiss:
        ordits=[sum([sum([kffis[2][nki][rs[nbid[0]][1],rs[nbid[1]][1]]*Pm[nbid[0],nbid[1]]*(Nsl/Nst)*e**(1.j*(np.dot(ltc.pos(rs[nbid[0]],ltype)-RDV[nbid[0],nbid[1]]-r0c,kffis[1][nki]+kffis[3][nki]*kffis[0])-np.dot(ltc.pos(rs[nbid[0]],ltype)-r0c,kffis[1][nki]))) for nbid in nbids]) for nbids in nbidss]) for nki in range(len(kffis[1]))]
        ordifs=[np.dot(np.array(ordits),np.array(ff)).imag for ff in kffis[4]]
        ordifss+=[ordifs]
    return ordrfss,ordifss


def orderinvfourier(kffs,ltype,rs,r0c,NB,RDV,Nrfl,oritype):
    [Nr,Nfl]=Nrfl
    Nst=tb.statenum(Nrfl)
    Nsl=ltc.slnum(ltype)
    if(oritype=='r'):ori=1.
    elif(oritype=='i'):ori=1.j
    P=sum([kffs[4][0][nk]*ori*np.array([[(NB[rid0,rid1]<=1)*kffs[2][nk][rs[rid0][1],rs[rid1][1]]*(Nsl/Nst)*e**(1.j*(-np.dot(ltc.pos(rs[rid0],ltype)-RDV[rid0,rid1]-r0c,kffs[1][nk]+kffs[3][nk]*kffs[0])+np.dot(ltc.pos(rs[rid0],ltype)-r0c,kffs[1][nk]))) for rid1 in range(Nr)] for rid0 in range(Nr)]) for nk in range(len(kffs[1]))])
    return P


def neighbors(NB,RDV):
    nb1idss=[np.argwhere(NBrid==1).flatten() for NBrid in NB]
    FFi=np.array([[(NB[rid0,rid1]==1)*((-1)**round(abs(np.dot(RDV[rid0,rid1],np.array([0.,1.,0.]))))) for rid1 in range(NB.shape[1])] for rid0 in range(NB.shape[0])])
    return nb1idss,FFi


def orderdev(Pm,rs,NB,nb1idss,FFi):
    Nnb1=len(nb1idss[0])
    dordrrs=np.array([(1./2.)*(Pm[rid,rid]-(1./Nnb1)*sum([Pm[rid1,rid1] for rid1 in nb1idss[rid]]))*((-1)**rs[rid][1]) for rid in range(len(nb1idss))]).real
    avgdordr,sgdordr=np.average(dordrrs).real,np.std(dordrrs)
    Jnb1m=(Pm*FFi).imag
    dordirs=np.array([(sum([Jnb1m[rid,rid1] for rid1 in nb1idss[rid]]))*((-1)**(1+rs[rid][1])) for rid in range(len(nb1idss))])
    avgdordi,sgdordi=np.average(dordirs),np.std(dordirs)
    return avgdordr,avgdordi,sgdordr,sgdordi,dordrrs,dordirs


def chernnumfun(H,ltype,rs,RDV,Nrfl,nf,rucs,RUCRP,ks,dks,tobdg=False):
    # Get the momentum-space Hamiltonian.
    Hk=lambda k:bdth.ftham(k,H,Nrfl,RDV,rucs,RUCRP,tobdg=tobdg)
    dBfs=[bdth.berrycurv(k,Hk,dks,nf,tobdg=tobdg)[0] for k in ks]
    Ch=(1./(2*pi))*sum(dBfs)
    return Ch


def ybdmat(RDV,Nrfl):
    nvy=np.array([0.,1.,0.])
    Yin=np.array([[np.dot(nvy,RDV[rid0,rid1]) for rid1 in range(Nrfl[0]) for fl1 in range(Nrfl[1])] for rid0 in range(Nrfl[0]) for fl0 in range(Nrfl[1])])
    return Yin


def ycurrent(Pm,H0t,Yin,Nst):
    yj=-(1./Nst)*(np.sum(Pm*H0t*Yin)).imag
    return yj


def timeavg(tts,data,ttc,ttavg,toavgall=True):
    nttavg=round(ttavg/(tts[1]-tts[0]))
    def avg(dat,ntt):
        if(toavgall):return np.average(np.array(dat[0:ntt]))
        else:return np.average(np.array(dat[ntt-min(ntt,nttavg):ntt+1]))
    return [[avg(dat,ntt) for ntt in range(len(tts))] for dat in data]


class CustomFormatter(mplb.ticker.ScalarFormatter):
    def __init__(self, digits=2):
        super().__init__(useOffset=False, useMathText=True)
        self.set_scientific(True)
        self.set_powerlimits((-2, 2))
        self.digits = digits

    def _set_format(self):
        self.format = f'%.{self.digits}f'


def plotevol(tts,data,ylabel,datanum=1,labels=[],titlet='',ttmin=-1,ttmax=-1,dlims=False,tosave=False,filetfig='',tosmooth=False,tosety=False,toplqt=False,ytickst=[],tobgc=False):
    '''
    Plot the evolution of the energy.
    '''
    if(ttmax!=-1 and ttmax<tts[-1]):
        ttst=np.array(tts)
        ttminn,ttmaxn=np.argwhere(abs(ttst-ttmin)<1e-8)[0][0],np.argwhere(abs(ttst-ttmax)<1e-8)[0][0]
        tts=tts[ttminn:ttmaxn+1]
        for n in range(len(data)):data[n]=data[n][ttminn:ttmaxn+1]
    plt.rcParams.update({'font.size':30})
#    plt.rcParams['figure.figsize']=[10,8]
#    fig,ax=plt.subplots()
    fig=plt.figure(figsize=(8,8))
    if(datanum==1):gs=mplb.gridspec.GridSpec(1,1,left=0.39,right=0.95,top=0.68,bottom=0.12)
#    if(datanum==1):gs=mplb.gridspec.GridSpec(1,1,left=0.6,right=0.95,top=0.68,bottom=0.12)
#    if(datanum==1):gs=mplb.gridspec.GridSpec(1,1,left=0.65,right=0.95,top=0.68,bottom=0.12)
    elif(datanum==2):gs=mplb.gridspec.GridSpec(1,1,left=0.3,right=0.95,top=0.68,bottom=0.12)
#    elif(datanum==2):gs=mplb.gridspec.GridSpec(1,1,left=0.25,right=0.95,top=0.68,bottom=0.12)
    ax=fig.add_subplot(gs[0,0])
#    if(tobgc):ax.axvspan(0,50,facecolor='red',alpha=0.1,label='_nolegend_')
    if(tobgc):ax.axvspan(0,50,facecolor=[92/255,64/255,64/255],alpha=0.075,label='_nolegend_')
    if(toplqt):
        plt.axhline(y=1,color='r',linestyle='--')
        plt.axhline(y=-1,color='b',linestyle='--')
    # Plot the spectrum.
    if(tosmooth):
        tts0=tts
        tts=np.linspace(tts0[0],tts0[-1],10*len(tts0))
    ddlims=dlims[1]-dlims[0]
    zorders=len(data)*[2]
    if(datanum==1):
        colors=[[0.8500,0.3250,0.0980],[0.4660,0.6740,0.1880],[0,0.4470,0.7410]]
        if(len(data)>3):
            colors+=(np.array(3*[[1.,1.,1.]])-0.5*(np.array(3*[[1.,1.,1.]])-np.array(colors))).tolist()
            zorders=[2,2,2]+(len(data)-3)*[1.9]
        ylims=[dlims[0]-0.35*ddlims,dlims[1]+0.22*ddlims]
    elif(datanum==2):
        cmap=mplb.colormaps['coolwarm']
        colors=cmap(np.linspace(1.,0.,len(data)))
        ylims=[dlims[0]-0.65*ddlims,dlims[1]+0.28*ddlims]
    for ndata in range(len(data)):
        dat=data[ndata]
        if(tosmooth):
            spl=make_interp_spline(tts0,dat,k=2)
            dat=spl(tts)
        plt.plot(tts,dat,linewidth=2.,color=colors[ndata],zorder=zorders[ndata])
    plt.gca().autoscale()
    # Set the label of the axes.
    plt.ylabel(ylabel)
    ymax=np.max(np.abs(np.array(data)))
    plt.ylim(ylims[0],ylims[1])
    dylims=ylims[1]-ylims[0]
    if(len(ytickst)==0):
        if(dylims>1e-3 and dylims<1e-1):
            if(dylims>3e-2):dtky=1e-2
            elif(dylims>1.8e-2):dtky=5e-3
            elif(dylims>1.2e-2):dtky=3e-3
            elif(dylims>6e-3):dtky=2e-3
            elif(dylims>1e-3):dtky=1e-3
            ax.yaxis.set_major_locator(mplb.ticker.MultipleLocator(dtky))
            ytkst=ax.get_yticks()
            ytks=[]
            for ytk in ytkst:
                if(ytk>ylims[0] and ytk<dlims[1]+0.1*ddlims):ytks+=[ytk]
            ax.set_yticks(ytks)
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter('${%.3f}$'))
        '''
        # Apply formatter to y-axis
        formatter=CustomFormatter(digits=1)
        ax.yaxis.set_major_formatter(formatter)
        # Move offset text
        ax.yaxis.get_offset_text().set_position((0,1))
        ax.yaxis.get_offset_text().set_ha('right')
        ax.yaxis.get_offset_text().set_va('center')
        '''
    else:ax.set_yticks(ytickst[0],ytickst[1])
    if(toplqt):plt.ylim(-1.5,1.5)
    if(ttmax!=-1):plt.xlim(0.,ttmax)
    else:plt.xlim(tts[0],tts[-1])
    if(ttmax==50):ax.set_xticks([0,25,50])
    elif(datanum==1):ax.set_xticks([0,250,500])
    elif(datanum==2):ax.set_xticks([0,100,200,300,400,500])
    plt.xlabel('$t$')
    ax.text(0.5,0.92,titlet,horizontalalignment='center',verticalalignment='center',transform=ax.transAxes)
    if(len(labels)>0):
        if(datanum==1):
            lgd=plt.legend(labels[1],title=labels[0],loc='lower center',framealpha=1,bbox_to_anchor=(0.5,-0.02),ncol=3,handlelength=0.5,handletextpad=0.25,columnspacing=0.5)
            def legend_title_left(lgd):
                c=lgd.get_children()[0]
                title=c.get_children()[0]
                hpack=c.get_children()[1]
                c._children=[hpack]
                hpack._children=[title]+hpack.get_children()
            legend_title_left(lgd)
        elif(datanum==2):
            lgd=plt.legend(labels,title='$A_\mathrm{c}$',framealpha=1,loc='lower center',bbox_to_anchor=(0.5,-0.02),ncol=3,handlelength=0.5,handletextpad=0.25,columnspacing=0.5,labelspacing=0.25)
#            '''
            def legend_title_left(lgd):
                c=lgd.get_children()[0]
                title=c.get_children()[0]
                hpack=c.get_children()[1]
                c._children=[hpack]
                hpack._children=[title]+hpack.get_children()
            legend_title_left(lgd)
#            '''
    plt.gcf()
#    if(tosave==True):plt.savefig(filetfig,dpi=2000,bbox_inches='tight',pad_inches=0,transparent=True)
    if(tosave==True):plt.savefig(filetfig,dpi=2000,transparent=True)
    plt.show()

def fourierfreq(tts,data,omegatmax,nomegat=1000,tt0=0.,fittype='lin'):
    ntt0=round(tt0/(tts[1]-tts[0]))
    tts,data=(np.array(tts[ntt0:])-tt0).tolist(),[dat[ntt0:] for dat in data]
#    data=[(np.array(dat)-np.average(np.array(dat))).tolist() for dat in data]
    for ndata in range(len(data)):
        dat=data[ndata]
        if(fittype=='lin'):
            lrgs=linregress(tts,dat)
            a,b=lrgs.slope,lrgs.intercept
            lrgsl=a*np.array(tts)+b
            print('Linregress: slope =',a,', intercept =',b)
            data[ndata]=(np.array(dat)-lrgsl).tolist()
        elif(fittype=='exp'):
            lrgs=linregress(tts,dat)
            alrgs,blrgs=lrgs.slope,lrgs.intercept
            def fexp(tt,a,ttr,tau,b):return a*(e**(-(tt-ttr)/tau))+b
            acf,ttrcf,taucf,bcf=curve_fit(fexp,tts,dat,p0=np.array([alrgs,0,1,blrgs]),method='trf')[0]
            print('Curve fit: amplitude acf =',acf,', reference time ttrcf =',ttrcf,', relaxation time taucf=',taucf,', intercept bcf =',bcf)
            fexpl=[fexp(tt,acf,ttrcf,taucf,bcf) for tt in tts]
            data[ndata]=(np.array(dat)-np.array(fexpl)).tolist()
    print('data shape=',np.array(data).shape)
    omegats=np.linspace(0.,omegatmax,nomegat)
    Ntt=len(tts)
    ftass=[[abs((1./Ntt)*sum([dat[nt]*(e**(1.j*omegat*tts[nt])) for nt in range(Ntt)])) for omegat in omegats] for dat in data]
    return [omegats,ftass]


def plotftfreq(omegats,ftass,ylims=[],labels=[],titlet='',tosave=False,filetfig='',tosmooth=False,toinset=False):
    '''
    Plot the evolution of the energy.
    '''
    plt.rcParams.update({'font.size':30})
#    plt.rcParams['figure.figsize']=[10,8]
#    fig,ax=plt.subplots()
    fig=plt.figure(figsize=(8,8))
    gs=mplb.gridspec.GridSpec(1,1,left=0.25,right=0.95,top=0.68,bottom=0.12)
    ax=fig.add_subplot(gs[0,0])
    cmap=mplb.colormaps['coolwarm']
    colors=cmap(np.linspace(1.,0.,len(ftass)))
    for nftas in range(len(ftass)):plt.plot(omegats,ftass[nftas],linewidth=2.,color=colors[nftas])
    plt.gca().autoscale()
    # Set the label of the axes.
    plt.ylabel('$\Delta_2(\omega)$')
    if(len(ylims)>0):plt.ylim(ylims[0],ylims[1])
    dylims=ylims[1]-ylims[0]
    if(dylims>1e-4):
        if(dylims>3e-3):dtky=1e-3
        elif(dylims>1.8e-3):dtky=5e-4
        elif(dylims>1.2e-3):dtky=3e-4
        elif(dylims>6e-4):dtky=2e-4
        elif(dylims>1e-4):dtky=1e-4
        ax.yaxis.set_major_locator(mplb.ticker.MultipleLocator(dtky))
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
    # Apply formatter to y-axis
    formatter=CustomFormatter(digits=1)
    ax.yaxis.set_major_formatter(formatter)
    # Move offset text
    ax.yaxis.get_offset_text().set_position((0,1))
    ax.yaxis.get_offset_text().set_ha('right')
    ax.yaxis.get_offset_text().set_va('center')
#    ax.yaxis.set_major_formatter(mplb.ticker.FormatStrFormatter('%.4f'))
    plt.xlim(omegats[0],omegats[-1])
    plt.xlabel('$\omega$')
    ax.text(0.5,0.92,titlet,horizontalalignment='center',verticalalignment='center',transform=ax.transAxes)
    if(len(labels)>0):plt.legend(labels,title='$A_\mathrm{c}$',loc='upper right',bbox_to_anchor=(1.02,1.02),ncol=1,handlelength=0.5,handletextpad=0.25,labelspacing=0.25)
    if(toinset):
        ompeak002ntt=np.argmax(np.array(ftass[-1]))
#        ompeaks=[omegats[np.argmax(np.array(ftas))] for ftas in ftass]
        ompeaks=[omegats[round(0.5*ompeak002ntt):round(1.5*ompeak002ntt)][np.argmax(np.array(ftas[round(0.5*ompeak002ntt):round(1.5*ompeak002ntt)]))] for ftas in ftass]
        insetax=fig.add_axes([0.55,0.38,0.14,0.14],transform=ax.transAxes)
#        insetax=fig.add_axes([0.495,0.42,0.14,0.14],transform=ax.transAxes)
        insetax.plot([0.1,0.08,0.06,0.04,0.02],ompeaks,color=[0.4660,0.6740,0.1880],linestyle='--')
        insetax.scatter([0.1,0.08,0.06,0.04,0.02],ompeaks,color=[0.4660,0.6740,0.1880])
        insetax.set_xlim(0.,0.1)
        insetax.set_xticks([0.,0.1],['0.0','0.1'])
        insetax.set_ylim(min(ompeaks),max(ompeaks))
        # Apply formatter to y-axis
        formatter=CustomFormatter(digits=2)
        insetax.yaxis.set_major_formatter(formatter)
        # Move offset text
        insetax.yaxis.get_offset_text().set_position((0,1))
        insetax.yaxis.get_offset_text().set_ha('right')
        insetax.yaxis.get_offset_text().set_va('bottom')
        insetax.set_xlabel('$A_\mathrm{c}$')
        insetax.set_ylabel('$\omega$')
#    ax.text(0.5,0.77,'Peak',horizontalalignment='center',verticalalignment='center',transform=ax.transAxes)
    plt.gcf()
#    if(tosave==True):plt.savefig(filetfig,dpi=2000,bbox_inches='tight',pad_inches=0,transparent=True)
    if(tosave==True):plt.savefig(filetfig,dpi=2000,transparent=True)
    plt.show()






