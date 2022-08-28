from turtle import forward
import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn import Conv2d,MaxPool2d,BatchNorm2d,ReLU,Linear,Sigmoid,Dropout


class ResBlock(Module):
    def __init__(self,in_channels,out_channels,drop_prob) -> None:
        super().__init__()
        self.bn0 = BatchNorm2d(in_channels)
        self.relu0 = ReLU()
        self.conv0 = Conv2d(in_channels,out_channels,3,padding="same")
        self.dropout0 = Dropout(drop_prob)

        self.bn1 = BatchNorm2d(out_channels)
        self.relu1 = ReLU()
        self.conv1 = Conv2d(out_channels,out_channels,3,padding="same")
        self.dropout1 = Dropout(drop_prob)

        self.attn_bn0 = BatchNorm2d(in_channels)
        self.attn_relu0 = ReLU()
        self.attn_conv0 = Conv2d(in_channels,out_channels,3,padding="same")
        self.sigmoid = nn.Sigmoid()

    def forward(self,input):
        x = input

        x = self.bn0(x)
        x = self.relu0(x)
        x = self.conv0(x)

        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)

        # attn = self.attn_conv0(self.attn_relu0(self.attn_bn0(input)))
        # attn = self.sigmoid(attn)

        # x = x * attn

        return x + input


class ResNet_Fig(Module):
    def __init__(self,depth,drop_prob) -> None:
        super().__init__()
        self.depth = depth
        
        self.batchnorm0 = nn.BatchNorm2d(3)
        self.relu0 = nn.ReLU()
        self.conv0 = nn.Conv2d(3,32,5,padding="same")
        self.maxpool0 = MaxPool2d(2,2)

        self.resblocks = nn.ModuleList([ResBlock(32,32,drop_prob) for i in range(self.depth)])

        self.linear = Linear(15*10*32,30*20)
        self.sigmoid = Sigmoid()

    def forward(self,input):
        x = input

        x = self.conv0(self.relu0(self.batchnorm0(x)))
        x = self.maxpool0(x)

        for resblock in self.resblocks:
            x = resblock(x)

        x = x.reshape(x.shape[0],15*10*32)
        x = self.linear(x)
        x = x.reshape(x.shape[0],30,20)

        return x


class ResNet(Module):
    def __init__(self,depth,drop_prob) -> None:
        super().__init__()
        self.depth = depth

        self.batchnorm0 = nn.BatchNorm2d(3)
        self.relu0 = nn.ReLU()
        self.conv0 = nn.Conv2d(3,8,5,padding="same")
        self.maxpool0 = MaxPool2d(2,2)

        self.resblocks = nn.ModuleList([ResBlock(8,8,drop_prob) for i in range(self.depth)])

        self.batchnorm1 = nn.BatchNorm1d(15*10*8)
        self.relu1 = nn.ReLU()
        self.linear1 = Linear(15*10*8,1)

        self.sigmoid = Sigmoid()

    def forward(self,input):
        x = input

        x = self.conv0(self.relu0(self.batchnorm0(x)))
        x = self.maxpool0(x)

        for resblock in self.resblocks:
            x = resblock(x)

        x = x.reshape(x.shape[0],15*10*8)
        x = self.linear1(self.relu1(self.batchnorm1(x)))
        return x