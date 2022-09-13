import torch

def MAPE(target,output):
    delta = torch.abs(target - output)
    result = torch.mean(delta / target)
    return result

def MAE(target,output):
    delta = torch.abs(target - output)
    result = torch.mean(delta)
    return result