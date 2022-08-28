from turtle import width
import torch
import torch.nn as nn
from torch.nn import Conv2d,Module



class BN_Act_Conv(Module):
    def __init__(self,in_channels,out_channels,keep_prob) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv = Conv2d(in_channels,out_channels,3,padding="same")
        self.dropout = nn.Dropout(keep_prob)

    def forward(self,input):
        x = self.bn(input)
        x = self.relu(x)
        x = self.conv(x)
        x = self.dropout(x)
        return x

class DenseBlock(Module):
    def __init__(self,layers,in_channels,growth,keep_prob) -> None:
        super().__init__()
        self.layers = layers
        self.module_list = nn.ModuleList()
        for i in range(layers):
            channels = in_channels + growth * i
            self.module_list.append(BN_Act_Conv(channels,growth,keep_prob))

    def forward(self,input):
        x = input
        for i in range(self.layers):
            result = self.module_list[i](x)
            x = torch.cat([x,result],dim=1)
        return x



class DenseNet(Module):
    def __init__(self,height,width,layers,growth,scale,keep_prob) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.growth = growth
        self.scale = scale
        self.keep_prob = keep_prob

        self.conv0 = Conv2d(3,16,2,padding="same")
        in_channels = 16

        self.denseblock1 = DenseBlock(layers,in_channels,growth,keep_prob)
        in_channels += growth * layers
        self.bn_act_conv_1 = BN_Act_Conv(in_channels,in_channels,keep_prob)
        self.avgpool = nn.AvgPool2d(2,2)
        self.height //= 2
        self.width //= 2

        self.denseblock2 = DenseBlock(layers,in_channels,growth,keep_prob)
        in_channels += growth * layers
        self.bn_act_conv_2 = BN_Act_Conv(in_channels,in_channels,keep_prob)

        self.linear = nn.Linear(in_channels * self.width * self.height,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,input):
        height,width = input.shape[2],input.shape[3]
        batch_size = input.shape[0]
        x = self.conv0(input)

        x = self.denseblock1(x)
        x = self.bn_act_conv_1(x)
        x = self.avgpool(x)

        x = self.denseblock2(x)
        x = self.bn_act_conv_2(x)

        x = x.reshape(batch_size,-1)
        x = self.linear(x)
        x = self.sigmoid(x)
        # x = x.reshape(batch_size,height,width)

        return x * self.scale




