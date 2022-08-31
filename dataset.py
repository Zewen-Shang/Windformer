from cProfile import label
import enum
from operator import concat
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

import pdb

BASE_URL = "../"

def generate_img(target_map,save_fig=False):
    # feature_names = ["speed_x","speed_y","temperature","pressure","density","speed"]
    feature_num = 6

    ids_map = np.load(BASE_URL + "home/map%d.npy"%target_map,allow_pickle=True)
    feature_data = np.zeros((feature_num,30,20,420769))

    for i in range(ids_map.shape[0]):
        for j in range(ids_map.shape[1]):
            # if(i == 15 and j == 10):
            #     print(ids_map[i,j])
            turbine_data = np.load(BASE_URL + "input/input%d/%d.npy"%(target_map,ids_map[i,j]),allow_pickle=True)
            feature_data[:,i,j] = turbine_data.transpose()

    # if save_fig:
    #     for i,feature_name in enumerate(feature_names):
    #         sns.kdeplot(data=feature_data[i].flatten()[:100000])
    #         plt.show()
    #         plt.savefig("./fig/" + feature_name + ".jpg")
    #         plt.cla()

    np.save(BASE_URL + "home/npy/result/img_data.npy",feature_data[5:6])
    # plt.imshow(target_data[:,:,:].std(axis=2))
    # plt.savefig("./std.jpg")

def generate_seq(save_fig=False):
    feature_names = ["speed_x","speed_y","temperature","pressure","density","speed"]
    feature_num = len(feature_names)
    
    ids = np.load("./seq_ids.npy")
    feature_data = np.ones((feature_num,ids.shape[0],105120))
    
    for i,feature_name in enumerate(feature_names):
        for j,id in enumerate(ids.tolist()):
            data_item = np.load("./npy/2010/11158-12212/%d.npy"%id,allow_pickle=True).item()[feature_name]
            # data_item = np.concatenate([data_item,np.load("./npy/2011/11158-12212/%d.npy"%id,allow_pickle=True).item()[feature_name]])
            feature_data[i,j] = data_item

    np.save("./npy/result/seq_data",feature_data)

"""
seasons : [0,1,2,3],可选
"""
def get_dataset_seq(target_pos,seasons):
    # seq_data (feature_num,drive_num,105120 * 2) 
    seq_data = np.load("./npy/result/seq_data.npy")
    timestep_num = seq_data.shape[2] // 2
    for i in range(seq_data.shape[0]-1):
        seq_data[i] = (seq_data[i] - seq_data[i].mean()) / seq_data[i].std()
    dataset = []
    for i in seasons:
        for j in range(int(i / 4 * timestep_num),int((i+1) / 4 * timestep_num - 11)):
            data_item = [torch.from_numpy(seq_data[:,:,j:j+6]).to(dtype=torch.float),torch.tensor(seq_data[-1,target_pos,j+11]).to(dtype=torch.float)]
            dataset.append(data_item)
    for i in seasons:
        for j in range(int((i+4) / 4 * timestep_num),int((i+5) / 4 * timestep_num - 11)):
            data_item = [torch.from_numpy(seq_data[:,:,j:j+6]).to(dtype=torch.float),torch.tensor(seq_data[-1,target_pos,j+11]).to(dtype=torch.float)]
            dataset.append(data_item)
    return dataset

    
def get_dataset_img(target_pos,window_size,predict_steps,seasons,debug=False):
    # feature_data (feature_num,30,20,105120)
    feature_data = np.load(BASE_URL + "home/npy/result/img_data.npy")
    if debug:
        feature_data = feature_data[0:200]
    for i in range(feature_data.shape[0]-1):
        feature_data[i] = (feature_data[i] - feature_data[i].mean()) / feature_data[i].std() / 1e7
        feature_data[i] = feature_data[i] * 0
    dataset = []

    years = 2
    time_steps = feature_data.shape[-1]
    year_steps = time_steps // years
    season_steps = year_steps // 4

    for year in range(years):
        year_start = int(year / years * year_steps)
        year_end = year_start + year_steps
        for season in seasons:
            season_start = year_start + int(season / 4 * season_steps)
            season_end = season_start + season_steps
            for i in range(season_start,season_end-(window_size+predict_steps-1)):
                data_item = [feature_data[:,:,:,i:i+window_size],
                np.array(feature_data[-1,target_pos[0],target_pos[1],i+(window_size+predict_steps-1)]).reshape(1,1)]
                dataset.append(data_item)
    return dataset
