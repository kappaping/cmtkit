## Main function

from math import *
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../lattice')
import lattice as ltc
import bandtheory as bdth
import bandstructure as bdst


ltype='tr'
print(ltc.ltcname(ltype))
uctype=111
Nk=100
ks=bdst.brillouinzone(ltype,uctype,Nk)
kxs,kys=[ks[n][0] for n in range(len(ks))],[ks[n][1] for n in range(len(ks))]
plt.scatter(kxs,kys)
plt.xlim(-5.,5.)
plt.ylim(-5.,5.)
plt.show()
