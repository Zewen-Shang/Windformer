import torch
import torch.nn as nn
from einops import rearrange

class Persist(nn.Module):
    def __init__(self,predict_steps) -> None:
        super().__init__()
        self.predict_steps = predict_steps

    def forward(self,x):
        # x (batch_size,input_steps,in_channels,image_shape)

        (batch_size,input_steps,in_channels),image_shape = x.shape[0:3],x.shape[3:5]
        
        #target (batch_size,1,*image_shape)
        target = x[:,-1:,-1]
        output = target.repeat(1,self.predict_steps,1,1)
        return output

