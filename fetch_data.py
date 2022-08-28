from pywtk.wtk_api import get_nc_data_from_url,get_nc_data
import pandas
import numpy as np

WTK_URL = "https://f9g6p4cbvi.execute-api.us-west-2.amazonaws.com/prod"
MET_DIR="/home/tju/test3/local-data-dir/met_data/"
#09-12 420769

id_maps = np.load("./map1.npy")
id_maps = np.unique(id_maps.flatten())

for i,site_id in enumerate(id_maps):
    # if(i < 17):
    #     continue
    print(i)
    start = pandas.Timestamp('2009-01-01', tz='utc')
    end = pandas.Timestamp('2011-01-01', tz='utc')
    utc = True
    attributes = ["wind_direction", "wind_speed", "temperature", "pressure", "density"]
    # met_data = get_nc_data_from_url(WTK_URL+"/met", str(site_id), start, end, attributes, utc=utc)
    met_data = get_nc_data(str(site_id),start,end,attributes,nc_dir=MET_DIR)
    met_data = np.array(met_data)
    rad = np.deg2rad(met_data[:,0])
    sin,cos = np.sin(rad),np.cos(rad)
    speed_x,speed_y = met_data[:,1] * cos,met_data[:,1] * sin
    result = np.zeros((met_data.shape[0],6))
    result[:,:3] = met_data[:,2:5]
    result[:,3],result[:,4],result[:,5] = speed_x,speed_y,met_data[:,1]
    np.save("./data/1/%d.npy"%site_id,result)

print()