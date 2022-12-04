import torch
from torch import nn
from einops import rearrange

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernels_per_layer=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels * kernels_per_layer,
            kernel_size=3,
            padding="same",
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(
            in_channels * kernels_per_layer, out_channels, kernel_size=1
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class SingledimConv(nn.Module):
    def __init__(self, height,width, input_steps, kernels_per_layer=1):
        super(SingledimConv, self).__init__()
        self.window = DepthwiseSeparableConv(
            input_steps, input_steps, kernels_per_layer=kernels_per_layer
        )
        self.bn_window = nn.BatchNorm2d(input_steps)

    def forward(self, input):
        # input (batch_size,height,width,input_steps)
        x_window = self.window(rearrange(input,"b h w win -> b win h w"))
        
        x_window = nn.functional.relu(self.bn_window(x_window))
        
        output = x_window.reshape(x_window.shape[0],-1)
        return output


class CNN(nn.Module):
    def __init__(self, num_features, img_shape, input_steps, hidden_neurons, kernels_per_layer=1, ):
        super(CNN, self).__init__()

        assert(num_features == 1)

        self.img_shape = img_shape
        self.input_steps = input_steps

        self.singledim = SingledimConv(
            img_shape[0],img_shape[1], input_steps, kernels_per_layer=kernels_per_layer,
        )

        self.merge = nn.Linear(
            img_shape[0] * img_shape[1] * input_steps, hidden_neurons)
        self.output = nn.Linear(hidden_neurons, 1)

    def forward(self, input):
        # input (batch_size,num_features,img_height,img_weight,input_steps)
        batch_size = input.shape[0]
        # x (batch_size,height,width,input_steps)
        x = input.reshape(batch_size,
                          self.img_shape[0],self.img_shape[1], self.input_steps)
        output = self.singledim(x)
        output = nn.functional.relu(self.merge(output))
        output = self.output(output)
        return output
