import numpy as np
import torch
from utils import *

BASE_URL = ".."


    
def get_dataset(target_map,input_steps,predict_steps,pad=3,debug=False):
    # data (times,num_features,30,20)
    data = np.load(BASE_URL + "/input/input%d/data%d.npy"%(target_map,target_map))
    time_steps,num_features,image_shape = data.shape[0],data.shape[1],(data.shape[2],data.shape[3])
    if debug:
        data = data[0:200]
    dataset = []
    # presist_output,presist_target = [],[]

    for i in range(0,time_steps-(input_steps+predict_steps)*pad-1):
        data_item = [data[i:i+(input_steps)*pad:pad,:,:,:],data[i+(input_steps)*pad:i+(input_steps+predict_steps)*pad:pad,-1,:,:]]
        dataset.append(data_item)
        # presist_output.append(np.tile(data[i+(input_steps-1)*pad,-1,:,:],(predict_steps,1,1)))
        # presist_target.append(data[i+(input_steps)*pad:i+(input_steps+predict_steps)*pad:pad,-1,:,:])
    
    # presist_target = np.stack(presist_target,axis=0)
    # presist_output = np.stack(presist_output,axis=0)

    # print("Presist MSE :  %f."%(((presist_target - presist_output)**2).mean()))
    
    return dataset


def test_persist(dataset_test,predict_steps):
    outputs,targets = [],[]
    for data_item in dataset_test:
        targets.append(data_item[1])
        outputs.append(data_item[0][-1:,-1,:,:].repeat(predict_steps,0))
    targets = np.stack(targets,axis=0)
    outputs = np.stack(outputs,axis=0)
    return MSE_np(outputs,targets),MAE_np(outputs,targets)

def dataset_norm(dataset_train,dataset_test):
    x_train = np.stack([item[0] for item in dataset_train],axis=0)
    B,I,F,Iw,Ih = x_train.shape
    mean,std = np.zeros((F)),np.zeros((F))

    for i in range(F):
        mean[i] = x_train[:,:,i,:,:].mean()
        std[i] = x_train[:,:,i,:,:].std()

    for i in range(len(dataset_train)):
        dataset_train[i][0] = (dataset_train[i][0] - mean.reshape(1,F,1,1)) / std.reshape(1,F,1,1)

    for i in range(len(dataset_test)):
        dataset_test[i][0] = (dataset_test[i][0] - mean.reshape(1,F,1,1))/std.reshape(1,F,1,1)

    return dataset_train,dataset_test

def dataset_np2torch(dataset):
    for i in range(len(dataset)):
        dataset[i][0] = torch.from_numpy(dataset[i][0]).to(dtype=torch.float)
        dataset[i][1] = torch.from_numpy(dataset[i][1]).to(dtype=torch.float)
    return dataset