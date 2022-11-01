import numpy as np


data = np.ones((3,4,5))

div3 = np.array([1,2,3])
div4 = np.array([1,2,3,4])
div5 = np.array([1,2,3,4,5])

div4 =  np.repeat(div4.reshape(1,4,1),[3,5],[0,20])

print()
