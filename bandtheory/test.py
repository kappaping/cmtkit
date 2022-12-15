from math import *
import numpy as np


x=[[] for n in range(3)]
y=[1,2,3]
[x[n].append(y[n]) for n in range(3)]
print(x)
