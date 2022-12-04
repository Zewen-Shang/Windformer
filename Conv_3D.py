import torch
from torch import nn
from einops import rearrange

class Conv_Block(nn.Module):
    def __init__(self,in_channels,out_channels) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels,out_channels,3,padding="same")
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

    def forward(self,x):
        return self.relu(self.conv(x))

class Conv_3D(nn.Module):
    def __init__(self,input_steps,predict_steps,num_features,dim,depth) -> None:
        """
        input : (B, input_steps, num_features, Iw, Ih)
        """
        super().__init__()

        self.input_steps,self.predict_steps,self.num_features = input_steps,predict_steps,num_features
        self.dim,self.depth = dim,depth

        self.conv1 = Conv_Block(num_features,dim)
        self.conv2 = Conv_Block(dim,dim)
        self.conv3 = Conv_Block(dim,dim)
        self.conv4 = Conv_Block(dim,1)
    
    def forward(self,x):

        B,I,F,_,_ = x.shape 
        x = rearrange(x,"B I F Iw Ih -> B F I Iw Ih")

        x = self.conv1(x)
        x = x + self.conv2(x)
        x = x + self.conv3(x)
        x = self.conv4(x)

        return rearrange(x,"B 1 I Iw Ih -> B I Iw Ih")
