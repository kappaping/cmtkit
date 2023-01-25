from math import *
import numpy as np
import json


x=[1,2]
print(x)

filet='data_test'
json.dump(x,filet)

y=json.load(filet)
print(y)
