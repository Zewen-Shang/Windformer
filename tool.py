from netCDF4 import Dataset
import numpy as np
import sys
import matplotlib.pyplot as plt
import numpy as np
# from mpl_toolkits.basemap import Basemap
from pandas import DataFrame
#数据读入
nc=Dataset('0.nc')

print(nc.variables.keys())
for var in nc.variables.keys():
    data=nc.variables[var][:].data
    print(var,data.shape)

print(nc.variables["wind_speed"])

import datetime
tstamp=(1478419200-613608 * 3600) #1900年1月1日零时距离1970年1月1日零时有613608个小时
date= datetime.datetime.utcfromtimestamp(tstamp)
print (date.strftime("%Y-%m-%d %H:%M:%S"))