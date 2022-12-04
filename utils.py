import torch
import numpy as np


def MSE_np(target,output):
    delta = (target-output)**2
    return delta.mean()


def MAPE(target,output):
    delta = torch.abs(target - output)
    result = torch.mean(delta / target)
    return result

def MAE_torch(target,output):
    delta = torch.abs(target - output)
    result = torch.mean(delta)
    return result

def MAE_np(target,output):
    delta = np.abs(target - output)
    result = delta.mean()
    return result