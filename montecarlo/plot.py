## Plot

import numpy as np
import matplotlib.pyplot as plt
import joblib




#Ts,chis,cTs=joblib.load('data_ising_16x16')
Ts,chis,cTs=joblib.load('data_ising_16x16')

#print('Ts = ',Ts)
#print('chis = ',chis)
#print('cTs = ',cTs)

plt.plot(Ts,chis,label='\u03C7')
plt.plot(Ts,cTs,label='c$_T$')
plt.xlabel('T')
plt.legend()
plt.show()
plt.savefig('test.png')

