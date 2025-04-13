## Main function

from math import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors
plt.rcParams['font.size']=18
plt.rcParams.update({'figure.autolayout': True})
import joblib


import sys
sys.path.append('../../cmt_code/lattice')
import lattice as ltc
sys.path.append('../../cmt_code/tightbinding')
import tightbinding as tb
import densitymatrix as dm
sys.path.append('../hartreefock')
import interaction as itn
import functionalrg as frg


# File name for writing out.
filetfig='../../figs/rgflow.pdf'
filet='../../data/test/test10'
#filet='../../data/functionalrg/triangular/nf_34_u0_40_nbmax_3_nqc_24'

phimaxss=joblib.load(filet)[3]

plt.rcParams.update({'font.size':30})
datas=phimaxss[0].T
plt.scatter(datas[0],datas[1],c='b')
datas=phimaxss[1].T
plt.scatter(datas[0],datas[1],c='r')
datas=phimaxss[2].T
plt.scatter(datas[0],datas[1],c='g')
plt.legend(['$P$','$C$','$D$'])
plt.xlabel('$T$')
plt.xscale('log')
plt.ylabel('max$(|\Phi_q|)$')
plt.ylim([-1,30])
#plt.yscale('log')
plt.gcf()
plt.savefig(filetfig,dpi=2000,bbox_inches='tight',pad_inches=0)
plt.show()

#'''
