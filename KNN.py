from unittest import result
import torch
from torch import nn
from torch.backends import cudnn
import matplotlib.pyplot as plt
import numpy as np 
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler

import json
from dataset import *

def MSE(target,output):
    target = np.array(target).squeeze()
    output = np.array(output).squeeze()
    delta = target - output
    result = np.mean(delta ** 2)
    return result

def MAE(target,output):
    target = np.array(target).squeeze()
    output = np.array(output).squeeze()
    delta = np.abs(target - output)
    result = np.mean(delta)
    return result

@torch.no_grad()
def test_model(model,dataloader,device):
    total_MSE,total_MAE = 0.,0.
    total_num = 0
    for imgs,targets in dataloader:
        imgs = imgs.to(device)
        targets = targets.to(device).squeeze()
        output = model(imgs).squeeze()
        mse = nn.MSELoss()(output,targets)
        
        mae = MAE(targets,output)
        total_MSE += mse * len(imgs)
        total_MAE += mae * len(imgs)
        total_num += len(imgs)
    return total_MSE / total_num,total_MAE / total_num


def test_time_displace(dataset_test):
    targets = []
    for data_item in dataset_test:
        targets.append(data_item[1])
    targets = torch.tensor(targets)
    output = targets.clone()
    output[6:] = targets[:len(targets)-6]
    for i in range(6):
        output[i] = output[0]
    return nn.MSELoss()(output,targets).item(),MAE(targets,output).item()


window_size = 6
season=[0,1,2,3]
# 23,15

for target_map in range(1,2):
    for predict_steps in [8,10]:
        # generate_img(target_map=target_map)

        dataset = get_dataset_img(target_pos=[15,10],window_size=window_size,predict_steps=predict_steps,seasons=season)
        # for i in range(len(dataset)):
        #     dataset[i][0] = dataset[i][0][-1:]

        X = [item[0][-1,15,10] for item in dataset]
        Y = [item[1].reshape(1) for item in dataset]

        cut_pos = int(0.75 * len(dataset))
        X_train,Y_train = X[:cut_pos],Y[:cut_pos]
        X_test,Y_test = X[cut_pos:],Y[cut_pos:]

        scaler = MinMaxScaler(feature_range =(-1,1)).fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # print(test_time_displace(dataset_test))

        model = KNeighborsRegressor(n_neighbors=10)

        model.fit(X_train,Y_train)

        import pickle
        with open('./model_new/KNN_%d_-1_%d.pkl'%(target_map,predict_steps), 'wb') as f:
            pickle.dump(model, f)

        Y_predict = model.predict(X_test)

        print("MSE:%f, MAE:%f"%(MSE(Y_predict,Y_test),MAE(Y_predict,Y_test)))

        np.save("./result/KNN_%d_-1_%d"%(target_map,predict_steps),Y_predict)