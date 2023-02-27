## Main function

from math import *
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../lattice')
import lattice as ltc
import bandtheory as bdth
import plotband as plbd


ltype='ka'
print(ltc.ltcname(ltype))
uctype=111
Nk=12
bzop=False
ks=bdth.brillouinzone(ltype,uctype,Nk,bzop)[0]
todata=True
dataks=[[ks[nk][0],ks[nk][1],0.] for nk in range(len(ks))]
plbd.plotbz(ltype,uctype,todata, dataks)
