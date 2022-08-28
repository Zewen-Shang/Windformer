from cProfile import label
import enum
from operator import concat
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

import pdb

BASE_URL = "../"

def generate_img(save_fig=False):
    # feature_names = ["speed_x","speed_y","temperature","pressure","density","speed"]
    feature_num = 6

    ids_map = np.load(BASE_URL + "home/map1.npy",allow_pickle=True)
    feature_data = np.zeros((feature_num,30,20,210241))

    for i in range(ids_map.shape[0]):
        for j in range(ids_map.shape[1]):
            # if(i == 15 and j == 10):
            #     print(ids_map[i,j])
            turbine_data = np.load(BASE_URL + "input/input1/%d.npy"%ids_map[i,j],allow_pickle=True)
            feature_data[:,i,j] = turbine_data.transpose()

    # if save_fig:
    #     for i,feature_name in enumerate(feature_names):
    #         sns.kdeplot(data=feature_data[i].flatten()[:100000])
    #         plt.show()
    #         plt.savefig("./fig/" + feature_name + ".jpg")
    #         plt.cla()

    np.save(BASE_URL + "home/npy/result/img_data.npy",feature_data)
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

    
def get_dataset_img(muti_output=False,target_pos=None):
    # feature_data (feature_num,30,20,105120)
    feature_data = np.load(BASE_URL + "home/npy/result/img_data.npy")
    for i in range(feature_data.shape[0]-1):
        feature_data[i] = (feature_data[i] - feature_data[i].mean()) / feature_data[i].std() / 1e7
        feature_data[i] = feature_data[i] * 0
    dataset = []
    for i in range(0,feature_data.shape[3]-13):
        # if i == 0:
        #     fig = plt.figure(1,(2,3),dpi=600)
        #     plt.axis("off")
        #     for j in range(feature_data.shape[0]):
        #         plt.imshow(feature_data[j,:,:,i])
        #         plt.savefig("./fig/heat_map/%d.jpg"%j,pad_inches=0.0,bbox_inches='tight')
        #         plt.show()
        if(muti_output):
            data_item = [torch.from_numpy(feature_data[:,:,:,i:i+6]).to(dtype=torch.float),
            torch.tensor(feature_data[-1,:,:,i+13]).to(dtype=torch.float)]
        else:
            data_item = [torch.from_numpy(feature_data[:,:,:,i:i+6]).to(dtype=torch.float),
            torch.tensor([feature_data[-1,target_pos[0],target_pos[1],i+13]]).to(dtype=torch.float)]
        dataset.append(data_item)
    return dataset
