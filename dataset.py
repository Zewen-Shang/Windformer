import numpy as np
import torch

BASE_URL = ".."


    
def get_dataset(target_map,window_size,predict_steps,debug=False):
    # data (times,feature_num,30,20)
    data = np.load(BASE_URL + "/input/input%d/data%d.npy"%(target_map,target_map))
    time_steps,feature_num,image_shape = data.shape[0],data.shape[1],(data.shape[2],data.shape[3])
    if debug:
        data = data[0:200]
    dataset = []

    for i in range(0,time_steps-(window_size+predict_steps-1)):
        data_item = [data[i:i+window_size,:,:,:],data[i+window_size:i+window_size+predict_steps,:,:,:]]
        dataset.append(data_item)

    mean,std = np.zeros((feature_num)),np.zeros((feature_num))
    # 归一化

    for i in range(feature_num):
        mean[i] = data[:,i,:,:].mean()
        std[i] = data[:,i,:,:].std()

    mean = mean.reshape(1,feature_num,1,1)
    std = std.reshape(1,feature_num,1,1)
    for i in range(len(dataset)):
        dataset[i][0] = (dataset[i][0] - mean) / std
    
    return dataset
