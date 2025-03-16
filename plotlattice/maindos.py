## Main function

from math import *
import numpy as np

import sys
sys.path.append('../lattice')
import lattice as ltc
import brillouinzone as bz
sys.path.append('../tightbinding')
import tightbinding as tb
import densitymatrix as dm
import bogoliubovdegennes as bdg
sys.path.append('../bandtheory')
import bandtheory as bdth
import plotband as plbd
import matplotlib.pyplot as plt
plt.rcParams['font.size']=18
plt.rcParams.update({'figure.autolayout': True})
from matplotlib.collections import LineCollection


# Lattice structure.
ltype='ka'
Nbl=[2,2,1]
rs,Nr=ltc.ltcsites(ltype,Nbl)
bc=1
filet='../../data/lattice/diamond/888_bc_1'
NB,RD,RDV=ltc.ltcpairdist(ltype,rs,Nbl,bc,toread=False,filet=filet)
# Flavor and state.
Nfl=1
Nrfl=[Nr,Nfl]
Nst=tb.statenum(Nrfl)

# Tight-binding Hamiltonian.
ts=[0.,-1.]
H=tb.tbham(ts,NB,Nfl)

# Set the unit cell with periodicity prds.
prds=[1,1,1]
rucs,RUCRP=bdth.ftsites(ltype,rs,prds)

# Get the momentum-space Hamiltonian.
Hk=lambda k:bdth.ftham(k,H,Nrfl,RDV,rucs,RUCRP)
Nkc=1200

filetfig='../../figs/testfigdos.pdf'
tosave=True

bzop=False
ks,dks=bz.listbz(ltype,prds,Nkc,bzop)
Nktot=len(ks)
print('Number of points =',Nktot)

ees=np.sort(np.array([np.linalg.eigvalsh(Hk(k)) for k in ks]).flatten()).tolist()
print('Finish energies.')

dees=np.linspace(ees[0],ees[-1],num=100)
dees[0]=dees[0]-1e-5
dees[-1]=dees[-1]+1e-5
nees=[]
nee=0
ndee=0
dee=dees[ndee]
for ee in ees:
    if(ee<dee):nee+=1
    else:
        nees+=[nee]
        nee=0
        ndee+=1
        dee=dees[ndee]
nees+=[nee]
print('Finish distribution.')

mu=0.+1e-5

def bandsegmentcolor(data0,data1,mu):
    # Determine the colors of a band segment [ee0,ee1].
    # Return green and blue below and above the chemical potential, respectively.
    if(data0<mu+1e-14 and data1<mu+1e-14):return 'g'
    else:return 'b'
cs=[bandsegmentcolor(dees[nee],dees[nee+1],mu) for nee in range(len(dees)-1)]
plt.rcParams.update({'font.size':30})
points=np.array([nees,dees]).T.reshape(-1,1,2)
segments=np.concatenate([points[:-1],points[1:]],axis=1)
# Add the collection of band segments to the plot.
lc=LineCollection(segments,colors=cs,linewidth=2)
plt.gca().add_collection(lc)
plt.gca().autoscale()

plt.axhline(y=mu,color='k',linestyle='--')

plt.xlim(0.,max(nees[0:-1])*1.1)
plt.xticks(ticks=[])
plt.xlabel('$D(E)$')
plt.ylim(dees[0]-0.04*(dees[-1]-dees[0]),dees[-1]+0.04*(dees[-1]-dees[0]))
plt.ylabel('$E_k$')
plt.gcf()
if(tosave==True):plt.savefig(filetfig,dpi=2000,bbox_inches='tight',pad_inches=0.01,transparent=True)
plt.show()




